import logging

# no need to set handler here, since we already set handler in root logger (utils.py), this child logger inherit that.
logger = logging.getLogger(__name__)
logger.info('Import sem.py')
import numpy as np
import tensorflow as tf
from scipy.special import logsumexp
from tqdm import tqdm
from .event_models import GRUEvent
from .utils import delete_object_attributes, unroll_data
import ray

# uncomment this line will generate weird error: cannot import GRUEvent...,
# actually because ray causes error while importing this file.
# ray.init()

# there are a ~ton~ of tf warnings from Keras, suppress them here
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
seed = int(os.environ.get('SEED', '1111'))
logger.info(f'Setting seed in sem.py, seed={seed}')
np.random.seed(seed)
tf.random.set_seed(seed)


class Results(object):
    """ placeholder object to store results """
    pass


class SEM(object):

    def __init__(self, lmda=1., alfa=10.0, kappa=1, threshold=0.4, trigger='pe', f_class=GRUEvent, f_opts=None):
        """
        Parameters
        ----------

        lmda: float
            sCRP stickiness parameter

        alfa: float
            sCRP concentration parameter

        f_class: class
            object class that has the functions "predict" and "update".
            used as the event model

        f_opts: dictionary
            kwargs for initializing f_class
        """
        self.lmda = lmda
        self.alfa = alfa
        self.kappa = kappa
        self.trigger = trigger
        self.threshold = threshold
        self.pe_window = []
        self.uncertainty_window = []
        # self.beta = beta

        if f_class is None:
            raise ValueError("f_model must be specified!")

        self.f_class = f_class
        self.f_class_remote = ray.remote(f_class)
        self.f_opts = f_opts

        # SEM internal state
        #
        self.n_clusters = 0  # maximum number of clusters (event types)
        self.c = np.array([])  # used by the sCRP prior -> running count of the clustering process
        self.c_eval = np.zeros(shape=(10000,))
        self.d = None  # dimension of scenes
        self.event_models = dict()  # event model for each event type
        self.model = None  # this is the tensorflow model that gets used, the architecture is shared while weights are specific

        self.x_prev = None  # last scene
        self.k_prev = None  # last event type
        self.x_curr = None  # current observed scene
        self.k_curr = None  # current highest posterior event type

        self.x_history = np.zeros(())

        # instead of dumping the results, store them to the object
        self.results = None

        # a general event model to initialize new events
        self.general_event_model = None
        self.general_event_model_x2 = None
        self.general_event_model_x3 = None
        self.general_event_model_yoke = None

    def pretrain(self, x, event_types, event_boundaries, progress_bar=True, leave_progress_bar=True):
        """
        Pretrain a bunch of event models on sequence of scenes X
        with corresponding event labels y, assumed to be between 0 and K-1
        where K = total # of distinct event types
        """
        assert x.shape[0] == event_types.size

        # update internal state
        k = np.max(event_types) + 1
        self._update_state(x, k)
        del k  # use self.k

        n = x.shape[0]

        # loop over all scenes
        if progress_bar:
            def my_it(l):
                return tqdm(range(l), desc='Pretraining', leave=leave_progress_bar)
        else:
            def my_it(l):
                return range(l)

        # store a compiled version of the model and session for reuse
        self.model = None

        for ii in my_it(n):

            x_curr = x[ii, :].copy()  # current scene
            k = event_types[ii]  # current event

            if k not in self.event_models.keys():
                # initialize new event model
                new_model = self.f_class(self.d, **self.f_opts)
                if self.model is None:
                    self.model = new_model.init_model()
                else:
                    new_model.set_model(self.model)
                self.event_models[k] = new_model

            # update event model
            if not event_boundaries[ii]:
                # we're in the same event -> update using previous scene
                assert self.x_prev is not None
                self.event_models[k].update(self.x_prev, x_curr, update_estimate=True)
            else:
                # we're in a new event -> update the initialization point only
                self.event_models[k].new_token()
                self.event_models[k].update_f0(x_curr, update_estimate=True)

            self.c[k] += 1  # update counts

            self.x_prev = x_curr  # store the current scene for next trial
            self.k_prev = k  # store the current event for the next trial

        self.x_prev = None  # Clear this for future use
        self.k_prev = None  #

    def _update_state(self, x, n_clusters=None):
        """
        Update internal state based on input data X and max # of event types (clusters) K
        """
        # get dimensions of data
        [n, d] = np.shape(x)
        if self.d is None:
            self.d = d
        else:
            assert self.d == d  # scenes must be of same dimension

        # get max # of clusters / event types
        if n_clusters is None:
            n_clusters = n
        self.n_clusters = max(self.n_clusters, n_clusters)

        # initialize CRP prior = running count of the clustering process
        if self.c.size < self.n_clusters:
            self.c = np.concatenate((self.c, np.zeros(self.n_clusters - self.c.size)), axis=0)
        assert self.c.size == self.n_clusters

    def _calculate_unnormed_sCRP(self, prev_cluster=None):
        # internal function for consistency across "run" methods

        # calculate sCRP prior
        prior = self.c.copy()
        # equating all visited clusters to the same value instead of the frequency of visits
        # Model Modification: comment for sCRP
        prior[prior > 0] = 1
        idx = len(np.nonzero(self.c)[0])  # get number of visited clusters

        # tan's code to correct when k is not None
        if idx < self.n_clusters:
            prior[idx] += self.alfa  # set new cluster probability to alpha
        # if idx <= self.k:
        #     prior[idx] += self.alfa  # set new cluster probability to alpha

        # add stickiness parameter for n>0, only for the previously chosen event
        if prev_cluster is not None:
            prior[prev_cluster] += self.lmda

        # prior /= np.sum(prior)
        return prior

    def init_general_models(self):
        if self.general_event_model is None:
            logger.info(f'Creating World model for initializations!')
            new_model = self.f_class(self.d, **self.f_opts)
            new_model.init_model()
            self.general_event_model = new_model
            new_model = None  # clear the new model variable (but not the model itself) from memory
        if self.general_event_model_x2 is None:
            logger.info(f'Creating x2 World model for initializations!')
            self.f_opts['n_hidden'] = int(self.f_opts['n_hidden'] * 2)
            new_model = self.f_class(self.d, **self.f_opts)
            new_model.init_model()
            self.general_event_model_x2 = new_model
            new_model = None  # clear the new model variable (but not the model itself) from memory
            self.f_opts['n_hidden'] = int(self.f_opts['n_hidden'] / 2)
        if self.general_event_model_x3 is None:
            logger.info(f'Creating x3 World model for initializations!')
            self.f_opts['n_hidden'] = int(self.f_opts['n_hidden'] * 3)
            new_model = self.f_class(self.d, **self.f_opts)
            new_model.init_model()
            self.general_event_model_x3 = new_model
            new_model = None  # clear the new model variable (but not the model itself) from memory
            self.f_opts['n_hidden'] = int(self.f_opts['n_hidden'] / 3)
        if self.general_event_model_yoke is None:
            logger.info(f'Creating yoke World model for initializations!')
            self.f_opts['n_hidden'] = int(self.f_opts['n_hidden'] * 3)
            new_model = self.f_class(self.d, **self.f_opts)
            new_model.init_model()
            self.general_event_model_yoke = new_model
            new_model = None  # clear the new model variable (but not the model itself) from memory
            self.f_opts['n_hidden'] = int(self.f_opts['n_hidden'] / 3)

    def update_current_event_model(self, event_boundary):
        """
        Update weights, variance, and history (update method) of the current event model
        Args:
            event_boundary: whether the current scene is the first scene of a new event

        Returns:

        """
        if event_boundary:
            # create a new token to avoid mixing with a distant past
            self.event_models[self.k_curr].new_token.remote()
            # re-add filler vector, need to update and recompute f0
            # Model Modification: uncomment to add f0
            # self.event_models[k].update_f0.remote(x_curr)
            # Model Modification: comment to add f0
            if self.x_prev is None:  # start of each run
                # assume that the previous scene is the same scene
                self.event_models[self.k_curr].update.remote(self.x_curr, self.x_curr)
            else:
                self.event_models[self.k_curr].update.remote(self.x_prev, self.x_curr)
        else:
            # we're in the same event -> update using previous scene
            assert self.x_prev is not None
            self.event_models[self.k_curr].update.remote(self.x_prev, self.x_curr)

    def update_generic_event_model(self):
        """
        Update weights, variance, and history (update method) of the generic event model
        Returns:

        """
        # for the world model, new token at the start of each new run
        if self.x_prev is None:  # start of each run
            self.general_event_model.new_token()
            # assume that the previous scene is the same scene, so that not using update_f0
            self.general_event_model.update(self.x_curr, self.x_curr)
        else:
            self.general_event_model.update(self.x_prev, self.x_curr)
        # # for yoke model, need to reset hidden units by creating a new token (no previous scenes).
        # if not event_boundary:
        #     self.general_event_model_yoke.update(self.x_prev, x_curr)
        # else:
        #     self.general_event_model_yoke.new_token()
        #     if self.x_prev is None:  # start of each run:
        #         self.general_event_model_yoke.update(x_curr, x_curr)
        #     else:
        #         self.general_event_model_yoke.update(self.x_prev, x_curr)

    def init_diagnostic_variables(self, x):
        """
        Initialize diagnostic variables based on dimensions of input data x
        :param x: input scenes (n, d)
        :param n: number of scenes
        Returns:

        """
        n = x.shape[0]
        self.results = Results()
        self.results.uncertainty = np.zeros(np.shape(x)[0])
        # initialize arrays to store results
        self.results.pe = np.zeros(np.shape(x)[0])
        self.results.pe_w = np.zeros(np.shape(x)[0])
        self.results.pe_w2 = np.zeros(np.shape(x)[0])
        self.results.pe_w3 = np.zeros(np.shape(x)[0])
        self.results.pe_yoke = np.zeros(np.shape(x)[0])
        self.results.x_hat = np.zeros(np.shape(x))
        self.results.x_hat_w = np.zeros(np.shape(x))
        self.results.x_hat_w2 = np.zeros(np.shape(x))
        self.results.x_hat_w3 = np.zeros(np.shape(x))
        self.results.after_relu = np.zeros((np.shape(x)[0], int(self.f_opts['n_hidden'])))
        self.results.after_relu_w = np.zeros((np.shape(x)[0], int(self.f_opts['n_hidden'])))
        self.results.after_relu_w2 = np.zeros((np.shape(x)[0], int(self.f_opts['n_hidden'] * 2)))
        self.results.after_relu_w3 = np.zeros((np.shape(x)[0], int(self.f_opts['n_hidden'] * 3)))
        self.results.boundaries = np.zeros((n,))
        self.results.log_like = np.zeros((n, self.n_clusters)) - np.inf
        self.results.log_prior = np.zeros((n, self.n_clusters)) - np.inf
        self.results.log_post = np.zeros((n, self.n_clusters)) - np.inf
        self.results.triggers = np.zeros((n,))

    def spawn_event_model(self):
        """
        Spawn a new event model, set its weights by the generic event model's weights
        Returns:
        new_model: the new event model
        """
        # This line trigger dynamic importing
        new_model = self.f_class_remote.remote(self.d, **self.f_opts)
        new_model.init_model.remote()
        new_model.do_reset_weights.remote()
        # if instead the following, model weights will be different from the above, which is weird!
        # model = ray.get(new_model.init_model.remote())
        # new_model.set_model.remote(model)

        # Model Modification: comment for random init instead of weights from generic model
        # set weights based on the general event model,
        # always use .model_weights instead of .model.get_weights() or .model.set_weights(...)
        # because .model_weights is guaranteed to be up-to-date.
        # logger.info('Set generic weights to new event schema')
        new_model.set_model_weights.remote(self.general_event_model.model_weights)
        return new_model

    def run(self, x, n_clusters=None, progress_bar=False, leave_progress_bar=True, minimize_memory=False, compile_model=True,
            train=True):
        """
        Parameters
        ----------
        x: N x D array of

        n_clusters: int
            maximum number of clusters

        progress_bar: bool
            use a tqdm progress bar?

        leave_progress_bar: bool
            leave the progress bar after completing?

        minimize_memory: bool
            function to minimize memory storage during running

        compile_model: bool (default = True)
            compile the stored model.  Leave false if previously run.
        train: bool (default=True)
            whether to train this video.

        Return
        ------
        post: n by k array of posterior probabilities

        """

        # update internal state
        self._update_state(x, n_clusters)
        self.init_general_models()
        self.init_diagnostic_variables(x)
        # these are special case variables to deal with the possibility the current event is restarted
        lik_restart_event = -np.inf
        repeat_prob = 0
        restart_prob = -np.inf

        # store the predicted vector and the after-relu vector for the current active event
        # x_hat_active, after_relu = None, None

        # pe_current, uncertainty_current = None, None  # for the active event, NOT the MAP event

        # this code just controls the presence/absence of a progress bar -- it isn't important
        if progress_bar:
            def my_it(l):
                return tqdm(range(l), desc='Run SEM', leave=leave_progress_bar)
        else:
            def my_it(l):
                return range(l)

        for ii in my_it(x.shape[0]):

            is_new_movie = (self.k_prev is None and self.x_prev is None) or (ii == 0)
            ### get the prior, likelihood, and posterior for all events and choose the best one
            self.x_curr = x[ii, :].copy()

            # get prior, prior always have one more non-zero element than #events,
            # because of the new event possibility
            prior = self._calculate_unnormed_sCRP(self.k_prev)  # prior: [0.01 0. 0. ...] (
            active = np.nonzero(prior)[0]  # active: [0]
            # if a new event was created in the previous step, then len(active) would increase by 1
            # and a new placeholder event model should be created in this step. (to calculate "likelihood").
            for count, k0 in enumerate(active):
                if k0 not in self.event_models.keys():
                    self.event_models[k0] = self.spawn_event_model()
                    logger.info(f"Spawned event model {k0}-th")
            # likelihood
            # lik = np.zeros(len(active))
            lik = np.full(len(active), -np.inf)
            # kwargs = {'x_curr': self.x_curr, 'x_prev': self.x_prev, 'k_prev': self.k_prev}
            jobs = []
            array_res = []
            # only calculate all other likelihoods when the threshold is exceeded or this is
            # the first scene of a new movie (self.k_prev and self.x_prev were reset by SEMContext)
            if is_new_movie:
                margin = self.threshold + 1
            else:
                # calculate the likelihood of the current event
                k0, pack = ray.get([self.event_models[self.k_prev].get_log_likelihood.remote(
                    k0=self.k_prev, k_prev=self.k_prev, x_curr=self.x_curr, x_prev=self.x_prev)])[0]
                after_relu, x_hat_active, lik[k0], lik_restart_event = pack
                pe_current = np.linalg.norm(self.x_curr - x_hat_active)
                uncertainty_current = ray.get([self.event_models[self.k_prev].get_uncertainty.remote(
                    self.x_curr, n_resample=16)])[0]
                if self.trigger == 'pe':
                    margin = pe_current - np.mean(self.pe_window)
                elif self.trigger == 'uncertainty':
                    # assert self.trigger == 'uncertainty', f'trigger must be pe or uncertainty, get {self.trigger}'
                    margin = uncertainty_current - np.mean(self.uncertainty_window)
                else:
                    margin = self.threshold + 1
            # if triggered iterate over all other event models (old and new, not current) and calculate likelihoods
            if margin > self.threshold:
            # if True:  # always trigger
                for count, k0 in enumerate(active):
                    # already calculated the likelihood for the active event, skip
                    if (not is_new_movie) and (k0 == self.k_prev):
                        continue
                    jobs.append(self.event_models[k0].get_log_likelihood.remote(
                        k0=k0, k_prev=self.k_prev, x_curr=self.x_curr, x_prev=self.x_prev))
                    # Chunking only constrain cpu usage, memory usage grows as
                    # #actors, self.f_class_remote.remote(self.d, **self.f_opts): 300MB per Actor.
                    # Actors will exit when the original handle to the actor is out of scope,
                    # thus, execute the jobs here instead of out of the loop
                    if (len(jobs) == 8) or (count == len(active) - 1):
                        assert count == k0, f"Sanity check failed, count={count} != k0={k0}"
                        array_res = array_res + ray.get(jobs)
                        jobs = []
                for (k0, pack) in array_res:
                    lik[k0] = pack
                self.results.triggers[ii] = True
            else:
                self.results.triggers[ii] = False
                # only restart the current event if the threshold is exceeded
                lik_restart_event = -np.inf
            # posterior
            # calculate posterior for all event models
            posterior = np.log(prior[:len(active)]) / self.d + lik
            # restarting the active event is an option only if this is not the first scene of a new movie
            if not is_new_movie:
                # is restart higher under the current event
                restart_prob = lik_restart_event + np.log(prior[self.k_prev] - self.lmda) / self.d
                repeat_prob = posterior[self.k_prev]
                posterior[self.k_prev] = np.max([repeat_prob, restart_prob])

            # choose the best (MAP) event
            if not train:
                # during evaluation, not consider creating a new event.
                self.k_curr = np.argmax(posterior[:-1])  # MAP cluster
            else:
                self.k_curr = np.argmax(posterior)  # MAP cluster
            logger.debug(f'\nEvent type {self.k_curr}')

            ### determine the type of event_boundary, then
            ### update event models' weights and internal SEM's states (activations, etc.)
            # determine whether there was a boundary, and which type
            event_boundary = (self.k_curr != self.k_prev) or ((self.k_curr == self.k_prev) and (restart_prob > repeat_prob))
            # 0=no_switch, 1=switch_old, 2=new, 3=restart_current
            if self.k_curr == len(active) - 1:
                self.results.boundaries[ii] = 2
            elif (self.k_curr == self.k_prev) and (restart_prob > repeat_prob):
                self.results.boundaries[ii] = 3
            else:
                self.results.boundaries[ii] = event_boundary  # could be 0 or 1

            if train:
                # update schema activations
                self.c[self.k_curr] += self.kappa  # update counts
                # train current event model, update history if there is an event boundary
                self.update_current_event_model(event_boundary)
                # train generic event model
                self.update_generic_event_model()
            else:
                self.c_eval[self.k_curr] += 1

            ### add pe or uncertainty to the running windows
            # if this is the first scene of a movie, then we assume that the previous event type
            # is the same as the MAP event and calculate PE and uncertainty for that event, treat x_curr as x_prev
            # otherwise, PE and uncertainty were calculated in the trigger section above
            if is_new_movie:
                _, x_hat_active = ray.get([self.event_models[self.k_curr].predict_next_generative.remote(self.x_curr)])[0]
                pe_current = np.linalg.norm(self.x_curr - x_hat_active)
                uncertainty_current = ray.get([self.event_models[self.k_curr].get_uncertainty.remote(
                    self.x_curr, n_resample=16)])[0]
            # if there were a boundary (either switching movies or real boundaries),
            # reset pe and uncertainty running windows
            if event_boundary:
                self.pe_window = []
                self.uncertainty_window = []
            self.pe_window.append(pe_current)
            self.uncertainty_window.append(uncertainty_current)

            ### store diagnostic information for the current scene, not affecting the models
            self.results.log_like[ii, :len(active)] = lik
            self.results.log_prior[ii, :len(active)] = np.log(prior[:len(active)]) / self.d
            self.results.log_post[ii, :len(active)] = posterior
            # can only make predictions if this is not the first scene of a new movie
            if not is_new_movie:
                self.results.x_hat[ii, :] = x_hat_active
                self.results.pe[ii] = pe_current
                self.results.uncertainty[ii] = uncertainty_current
                self.results.after_relu[ii, :] = after_relu
                # get predictions from world model to extract prediction error
                after_relu_w, x_hat_w = self.general_event_model.predict_next(self.x_prev)
                self.results.pe_w[ii] = np.linalg.norm(self.x_curr - x_hat_w)
                self.results.x_hat_w[ii, :] = x_hat_w
                self.results.after_relu_w[ii, :] = after_relu_w

            ### update scene and event type variables for next iteration
            self.x_prev = self.x_curr  # store the current scene for next trial
            self.k_prev = self.k_curr  # store the current event type for the next trial
            self.x_curr = None  # reset the current scene
            self.k_curr = None  # reset the current event type

        # Remove null columns to optimize memory storage, only for some arrays.
        self.results.log_like = self.results.log_like[:, np.any(self.results.log_like != -np.inf, axis=0)]
        self.results.log_prior = self.results.log_prior[:, np.any(self.results.log_prior != -np.inf, axis=0)]
        self.results.log_post = self.results.log_post[:, np.any(self.results.log_post != 0, axis=0)]
        # store more diagnostic information
        self.results.e_hat = np.argmax(self.results.log_post, axis=1)
        self.results.x = x
        # switching between video, not a real boundary
        self.results.boundaries[0] = 0
        self.results.c = self.c.copy()
        self.results.c_eval = self.c_eval.copy()
        self.results.Sigma = {i: ray.get(self.event_models[i].get_sigma.remote()) for i in self.event_models.keys()}
        self.results.weights = {i: ray.get(self.event_models[i].get_model_weights.remote()) for i in self.event_models.keys()}

    def update_single_event(self, x, update=True, save_x_hat=False):
        """

        :param x: this is an n x d array of the n scenes in an event
        :param update: boolean (default True) update the prior and posterior of the event model
        :param save_x_hat: boolean (default False) normally, we don't save this as the interpretation can be tricky
        N.b: unlike the posterior calculation, this is done at the level of individual scenes within the
        events (and not one per event)
        :return:
        """

        n_scene = np.shape(x)[0]

        if update:
            self.n_clusters += 1
            self._update_state(x, self.n_clusters)

            # pull the relevant items from the results
            if self.results is None:
                self.results = Results()
                post = np.zeros((1, self.n_clusters))
                log_like = np.zeros((1, self.n_clusters)) - np.inf
                log_prior = np.zeros((1, self.n_clusters)) - np.inf
                if save_x_hat:
                    x_hat = np.zeros((n_scene, self.d))
                    sigma = np.zeros((n_scene, self.d))
                    scene_log_like = np.zeros((n_scene, self.n_clusters)) - np.inf  # for debugging

            else:
                post = self.results.post
                log_like = self.results.log_like
                log_prior = self.results.log_prior
                if save_x_hat:
                    x_hat = self.results.x_hat
                    sigma = self.results.sigma
                    scene_log_like = self.results.scene_log_like  # for debugging

                # extend the size of the posterior, etc

                n, k0 = np.shape(post)
                while k0 < self.n_clusters:
                    post = np.concatenate([post, np.zeros((n, 1))], axis=1)
                    log_like = np.concatenate([log_like, np.zeros((n, 1)) - np.inf], axis=1)
                    log_prior = np.concatenate([log_prior, np.zeros((n, 1)) - np.inf], axis=1)
                    n, k0 = np.shape(post)

                    if save_x_hat:
                        scene_log_like = np.concatenate([
                            scene_log_like, np.zeros((np.shape(scene_log_like)[0], 1)) - np.inf
                        ], axis=1)

                # extend the size of the posterior, etc
                post = np.concatenate([post, np.zeros((1, self.n_clusters))], axis=0)
                log_like = np.concatenate([log_like, np.zeros((1, self.n_clusters)) - np.inf], axis=0)
                log_prior = np.concatenate([log_prior, np.zeros((1, self.n_clusters)) - np.inf], axis=0)
                if save_x_hat:
                    x_hat = np.concatenate([x_hat, np.zeros((n_scene, self.d))], axis=0)
                    sigma = np.concatenate([sigma, np.zeros((n_scene, self.d))], axis=0)
                    scene_log_like = np.concatenate([scene_log_like, np.zeros((n_scene, self.n_clusters)) - np.inf], axis=0)

        else:
            log_like = np.zeros((1, self.n_clusters)) - np.inf
            log_prior = np.zeros((1, self.n_clusters)) - np.inf

        # calculate un-normed sCRP prior
        prior = self._calculate_unnormed_sCRP(self.k_prev)

        # likelihood
        active = np.nonzero(prior)[0]
        lik = np.zeros((n_scene, len(active)))

        # again, this is a readout of the model only and not used for updating,
        # but also keep track of the within event posterior
        if save_x_hat:
            _x_hat = np.zeros((n_scene, self.d))  # temporary storre
            _sigma = np.zeros((n_scene, self.d))

        for ii, x_curr in enumerate(x):

            # we need to maintain a distribution over possible event types for the current events --
            # this gets locked down after termination of the event.
            # Also: none of the event models can be updated until *after* the event has been observed

            # special case the first scene within the event
            if ii == 0:
                event_boundary = True
            else:
                event_boundary = False

            # loop through each potentially active event model and verify
            # a model has been initialized
            for k0 in active:
                if k0 not in self.event_models.keys():
                    new_model = self.f_class(self.d, **self.f_opts)
                    if self.model is None:
                        self.model = new_model.init_model()
                    else:
                        new_model.set_model(self.model)
                    self.event_models[k0] = new_model

            ### ~~~~~ Start ~~~~~~~###

            ## prior to updating, pull x_hat based on the ongoing estimate of the event label
            if ii == 0:
                # prior to the first scene within an event having been observed
                k_within_event = np.argmax(prior)
            else:
                # otherwise, use previously observed scenes
                k_within_event = np.argmax(np.sum(lik[:ii, :len(active)], axis=0) + np.log(prior[:len(active)]))

            if save_x_hat:
                if event_boundary:
                    _x_hat[ii, :] = self.event_models[k_within_event].predict_f0()
                else:
                    _x_hat[ii, :] = self.event_models[k_within_event].predict_next_generative(x[:ii, :])
                _sigma[ii, :] = self.event_models[k_within_event].get_variance()

            ## Update the model, inference first!
            for k0 in active:
                # get the log likelihood for each event model
                model = self.event_models[k0]

                if not event_boundary:
                    # this is correct.  log_likelihood sequence makes the model prediction internally
                    # using predict_next_generative, and evaluates the likelihood of the prediction
                    lik[ii, k0] = model.log_likelihood_sequence(x[:ii, :].reshape(-1, self.d), x_curr)
                else:
                    lik[ii, k0] = model.log_likelihood_f0(x_curr)

        # cache the diagnostic measures
        log_like[-1, :len(active)] = np.sum(lik, axis=0)

        # calculate the log prior
        log_prior[-1, :len(active)] = np.log(prior[:len(active)])

        # # calculate surprise
        # bayesian_surprise = logsumexp(lik + np.tile(log_prior[-1, :len(active)], (np.shape(lik)[0], 1)), axis=1)

        if update:

            # at the end of the event, find the winning model!
            log_post = log_prior[-1, :len(active)] + log_like[-1, :len(active)]
            post[-1, :len(active)] = np.exp(log_post - logsumexp(log_post))
            k = np.argmax(log_post)

            # update the prior
            self.c[k] += n_scene
            # cache for next event
            self.k_prev = k

            # update the winning model's estimate
            self.event_models[k].update_f0(x[0])
            x_prev = x[0]
            for X0 in x[1:]:
                self.event_models[k].update(x_prev, X0)
                x_prev = X0

            self.results.post = post
            self.results.log_like = log_like
            self.results.log_prior = log_prior
            self.results.e_hat = np.argmax(post, axis=1)
            self.results.log_loss = logsumexp(log_like + log_prior, axis=1)

            if save_x_hat:
                x_hat[-n_scene:, :] = _x_hat
                sigma[-n_scene:, :] = _sigma
                scene_log_like[-n_scene:, :len(active)] = lik
                self.results.x_hat = x_hat
                self.results.sigma = sigma
                self.results.scene_log_like = scene_log_like

        return

    def init_for_boundaries(self, list_events):
        # update internal state

        k = 0
        self._update_state(np.concatenate(list_events, axis=0), k)
        del k  # use self.k and self.d

        # store a compiled version of the model and session for reuse
        if self.k_prev is None:
            # initialize the first event model
            new_model = self.f_class(self.d, **self.f_opts)
            self.model = new_model.init_model()

            self.event_models[0] = new_model

    def run_w_boundaries(self, list_events, progress_bar=True, leave_progress_bar=True, save_x_hat=False,
                         generative_predicitons=False, minimize_memory=False):
        """
        This method is the same as the above except the event boundaries are pre-specified by the experimenter
        as a list of event tokens (the event/schema type is still inferred).

        One difference is that the event token-type association is bound at the last scene of an event type.
        N.B. ! also, all of the updating is done at the event-token level.  There is no updating within an event!

        evaluate the probability of each event over the whole token


        Parameters
        ----------
        list_events: list of n x d arrays -- each an event


        progress_bar: bool
            use a tqdm progress bar?

        leave_progress_bar: bool
            leave the progress bar after completing?

        save_x_hat: bool
            save the MAP scene predictions?

        Return
        ------
        post: n_e by k array of posterior probabilities

        """

        # loop through the other events in the list
        if progress_bar:
            def my_it(iterator):
                return tqdm(iterator, desc='Run SEM', leave=leave_progress_bar)
        else:
            def my_it(iterator):
                return iterator

        self.init_for_boundaries(list_events)

        for x in my_it(list_events):
            self.update_single_event(x, save_x_hat=save_x_hat)
        if minimize_memory:
            self.clear_event_models()

    def clear_event_models(self):
        if self.event_models is not None:
            for _, e in self.event_models.items():
                e.clear()
                e.model = None

        self.event_models = None
        self.model = None
        tf.compat.v1.reset_default_graph()  # for being sure
        tf.keras.backend.clear_session()

    def clear(self):
        """ This function deletes sem from memory"""
        self.clear_event_models()
        delete_object_attributes(self.results)
        delete_object_attributes(self)


# @processify
def sem_run(x, sem_init_kwargs=None, run_kwargs=None):
    """ this initailizes SEM, runs the main function 'run', and
    returns the results object within a seperate process.

    See help on SEM class and on subfunction 'run' for more detail on the
    parameters contained in 'sem_init_kwargs'  and 'run_kwargs', respectively.

    Update (11/17/20): The processify function has been depricated, so this
    function no longer generates a seperate process.


    """

    if sem_init_kwargs is None:
        sem_init_kwargs = dict()
    if run_kwargs is None:
        run_kwargs = dict()

    sem_model = SEM(**sem_init_kwargs)
    sem_model.run(x, **run_kwargs)
    return sem_model.results


# @processify
def sem_run_with_boundaries(x, sem_init_kwargs=None, run_kwargs=None):
    """ this initailizes SEM, runs the main function 'run', and
    returns the results object within a seperate process.

    See help on SEM class and on subfunction 'run_w_boundaries' for more detail on the
    parameters contained in 'sem_init_kwargs'  and 'run_kwargs', respectively.

    Update (11/17/20): The processify function has been depricated, so this
    function no longer generates a seperate process.

    """

    if sem_init_kwargs is None:
        sem_init_kwargs = dict()
    if run_kwargs is None:
        run_kwargs = dict()

    sem_model = SEM(**sem_init_kwargs)
    sem_model.run_w_boundaries(x, **run_kwargs)
    return sem_model.results
