# If using multiprocessing, this module will be imported dynamically
print('Run sem.py')
import numpy as np

np.random.seed(1234)
import tensorflow as tf

tf.random.set_seed(1234)
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
import logging

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get('LOGLEVEL', logging.INFO))
# must have a handler, otherwise logging will use lastresort
c_handler = logging.StreamHandler()
LOGFORMAT = '%(name)s - %(levelname)s - %(message)s'
# c_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
c_handler.setFormatter(logging.Formatter(LOGFORMAT))
logger.addHandler(c_handler)
logger.debug('test sem')


class Results(object):
    """ placeholder object to store results """
    pass


class SEM(object):

    def __init__(self, lmda=1., alfa=10.0, kappa=1, f_class=GRUEvent, f_opts=None):
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
        # self.beta = beta

        if f_class is None:
            raise ValueError("f_model must be specified!")

        self.f_class = f_class
        self.f_class_remote = ray.remote(f_class)
        self.f_opts = f_opts

        # SEM internal state
        #
        self.k = 0  # maximum number of clusters (event types)
        self.c = np.array([])  # used by the sCRP prior -> running count of the clustering process
        self.d = None  # dimension of scenes
        self.event_models = dict()  # event model for each event type
        self.model = None  # this is the tensorflow model that gets used, the architecture is shared while weights are specific

        self.x_prev = None  # last scene
        self.k_prev = None  # last event type

        self.x_history = np.zeros(())

        # instead of dumping the results, store them to the object
        self.results = None

        # a general event model to initialize new events
        self.general_event_model = None

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

    def _update_state(self, x, k=None):
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
        if k is None:
            k = n
        self.k = max(self.k, k)

        # initialize CRP prior = running count of the clustering process
        if self.c.size < self.k:
            self.c = np.concatenate((self.c, np.zeros(self.k - self.c.size)), axis=0)
        assert self.c.size == self.k

    def _calculate_unnormed_sCRP(self, prev_cluster=None):
        # internal function for consistency across "run" methods

        # calculate sCRP prior
        prior = self.c.copy()
        # added on june 22 to test the case when old is not benefited
        prior[prior > 0] = 1
        idx = len(np.nonzero(self.c)[0])  # get number of visited clusters

        # tan's code to correct when k is not None
        if idx < self.k:
            prior[idx] += self.alfa  # set new cluster probability to alpha
        # if idx <= self.k:
        #     prior[idx] += self.alfa  # set new cluster probability to alpha

        # add stickiness parameter for n>0, only for the previously chosen event
        if prev_cluster is not None:
            prior[prev_cluster] += self.lmda

        # prior /= np.sum(prior)
        return prior

    def run(self, x, k=None, progress_bar=True, leave_progress_bar=True, minimize_memory=False, compile_model=True, train=True):
        """
        Parameters
        ----------
        x: N x D array of

        k: int
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
        self._update_state(x, k)
        del k  # use self.k and self.d

        n = x.shape[0]

        # initialize arrays
        # if not minimize_memory:
        post = np.zeros((n, self.k))
        pe = np.zeros(np.shape(x)[0])
        x_hat = np.zeros(np.shape(x))
        log_boundary_probability = np.zeros(np.shape(x)[0])
        # tan's code to encode types of boundaries for visualization
        boundaries = np.zeros((n,))

        # these are special case variables to deal with the possibility the current event is restarted
        lik_restart_event = -np.inf
        repeat_prob = -np.inf
        restart_prob = 0
        # tan's code to determine the predictive strength of the model
        restart_indices = []
        repeat_indices = []
        # frame_dynamics = dict(restart_lik=[], repeat_lik=[], new_lik=[], old_lik=[], restart_prior=[], repeat_prior=[],
        #                       new_prior=[], old_prior=[], post=[])

        #
        log_like = np.zeros((n, self.k)) - np.inf
        log_prior = np.zeros((n, self.k)) - np.inf

        # this code just controls the presence/absence of a progress bar -- it isn't important
        if progress_bar:
            def my_it(l):
                return tqdm(range(l), desc='Run SEM', leave=leave_progress_bar)
        else:
            def my_it(l):
                return range(l)

        for ii in my_it(n):

            x_curr = x[ii, :].copy()
            # parallel training a general event model
            if train:
                if self.general_event_model is None:
                    new_model = self.f_class(self.d, **self.f_opts)
                    if self.model is None:
                        self.model = new_model.init_model()
                    else:
                        new_model.set_model(self.model)
                    self.general_event_model = new_model
                    new_model = None  # clear the new model variable (but not the model itself) from memory
                # for the world model, new token at the start of each new run
                if self.x_prev is None:  # start of each run
                    self.general_event_model.new_token()
                    # self.general_event_model.update_f0(x_curr)
                    # assume that the previous scene is the same scene, so that not using update_f0
                    self.general_event_model.update(x_curr, x_curr)
                else:
                    self.general_event_model.update(self.x_prev, x_curr)

            # calculate sCRP prior
            prior = self._calculate_unnormed_sCRP(self.k_prev)
            # N.B. k_prev should be none for the first event if there wasn't pre-training

            # likelihood
            active = np.nonzero(prior)[0]
            lik = np.zeros(len(active))

            # store the predicted vector
            x_hat_active = None

            kwargs = {'x_curr': x_curr, 'x_prev': self.x_prev, 'k_prev': self.k_prev}
            jobs = []
            array_res = []
            for count, k0 in enumerate(active):
                if k0 not in self.event_models.keys():
                    # This line trigger dynamic importing
                    new_model = self.f_class_remote.remote(self.d, **self.f_opts)
                    new_model.init_model.remote()
                    new_model.do_reset_weights.remote()
                    # if instead the following, model weights will be different from the above, which is weird!
                    # model = ray.get(new_model.init_model.remote())
                    # new_model.set_model.remote(model)
                    self.event_models[k0] = new_model
                jobs.append(self.event_models[k0].get_likelihood.remote(k0, **kwargs))
                # Chunking only constrain cpu usage, memory usage grows as self.f_class_remote.remote(self.d, **self.f_opts)
                # 300MB per Actor.
                # Actors will exit when the original handle to the actor is out of scope,
                # thus, execute the jobs here instead of out of the loop
                if (len(jobs) == 16) or (count == len(active) - 1):
                    assert count == k0, f"Sanity check failed, count={count} != k0={k0}"
                    array_res = array_res + ray.get(jobs)
                    jobs = []
            for (k0, pack) in array_res:
                if k0 == self.k_prev:
                    x_hat_active, lik[k0], lik_restart_event = pack
                else:
                    lik[k0] = pack

            # determine the event identity (without worrying about event breaks for now)
            _post = np.log(prior[:len(active)]) / self.d + lik
            if ii > 0:
                # the probability that the current event is repeated is the OR probability -- but b/c
                # we are using a MAP approximation over all possibilities, it is a max of the repeated/restarted

                # is restart higher under the current event
                restart_prob = lik_restart_event + np.log(prior[self.k_prev] - self.lmda) / self.d
                repeat_prob = _post[self.k_prev]
                if restart_prob > repeat_prob:
                    logger.debug(f'\nlog_RESTART>log_repeat event_type {self.k_prev}')
                else:
                    logger.debug(f'\nlog_REPEAT>log_restart event_type {self.k_prev}')
                _post[self.k_prev] = np.max([repeat_prob, restart_prob])

                # readout probabilities, these are in raw scales so we can know the magnitude.
                # frame_dynamics['restart_lik'].append(lik_restart_event)
                # frame_dynamics['repeat_lik'].append(lik[self.k_prev])
                # # lik and prior and _post will be modified later, slicing to copy here
                # old_liks = [l for l in lik[:len(active) - 1]]
                # frame_dynamics['old_lik'].append(np.array(old_liks, dtype=float))
                # frame_dynamics['new_lik'].append(lik[len(active) - 1])
                #
                # frame_dynamics['restart_prior'].append(np.log(prior[self.k_prev] - self.lmda) / self.d)
                # frame_dynamics['repeat_prior'].append(np.log(prior[self.k_prev]) / self.d)
                # old_priors = [p for p in prior[:len(active) - 1]]
                # frame_dynamics['old_prior'].append(np.log(np.array(old_priors, dtype=float)) / self.d)
                # frame_dynamics['new_prior'].append(np.log(prior[len(active) - 1]) / self.d)
                # all_posteriors = [p for p in _post[:len(active)]]
                # frame_dynamics['post'].append(all_posteriors)
            logger.debug(f'\nlog_prior {np.log(prior[:len(active)]) / self.d}'
                         f'\nlog_lik {lik:}'
                         f'\nlog_post {_post:}')

            # get the MAP cluster and only update it
            k = np.argmax(_post)  # MAP cluster
            logger.debug(f'\nEvent type {k}')
            if k == self.k_prev:
                if restart_prob > repeat_prob:
                    restart_indices.append(ii)
                else:
                    repeat_indices.append(ii)
            if k != self.k_prev:
                logger.debug(f'Boundary Switching')
            # determine whether there was a boundary
            event_boundary = (k != self.k_prev) or ((k == self.k_prev) and (restart_prob > repeat_prob))
            # Add 2 and 3 to code for new_event and restarting
            if k == len(active) - 1:
                boundaries[ii] = 2
            elif (k == self.k_prev) and (restart_prob > repeat_prob):
                boundaries[ii] = 3
            else:
                boundaries[ii] = event_boundary  # could be 0 or 1
            # calculate the event boundary probability
            # _post is changed to calculate surprise
            _post[self.k_prev] = restart_prob
            # if not minimize_memory:
            log_boundary_probability[ii] = logsumexp(_post) - logsumexp(np.concatenate([_post, [repeat_prob]]))
            # calculate the probability of an event label, ignoring the event boundaries
            if self.k_prev is not None:
                # tan's version to keep it consistent w.r.t restart_prob and repeat_prob
                if restart_prob > repeat_prob:
                    prior[self.k_prev] -= self.lmda
                    lik[self.k_prev] = lik_restart_event
                _post[self.k_prev] = np.log(prior[self.k_prev]) / self.d + lik[self.k_prev]
                # _post[self.k_prev] = logsumexp([restart_prob, repeat_prob])
                # prior[self.k_prev] -= self.lmda / 2.
                # lik[self.k_prev] = logsumexp(np.array([lik[self.k_prev], lik_restart_event]))

                # now, the normalized posterior
                # if not minimize_memory:
                # Now, post should be equal to _post used to derive boundaries, no need for duplication
                # p = np.log(prior[:len(active)]) / self.d + lik
                # post[ii, :len(active)] = np.exp(p - logsumexp(p))
                # This is on a 0-1 scale
                post[ii, :len(active)] = np.exp(_post - logsumexp(_post))

                # this is a diagnostic readout and does not effect the model
                # these metrics don't distinguish between restart and repeat for k_prev,
                # but they are faster than frame_dynamics
                log_like[ii, :len(active)] = lik
                log_prior[ii, :len(active)] = np.log(prior[:len(active)]) / self.d

                # These aren't used again, remove from memory
                _post = None
                lik = None
                prior = None

            else:
                log_like[ii, 0] = 0.0
                log_prior[ii, 0] = self.alfa
                # if not minimize_memory:
                post[ii, 0] = 1.0

            if not minimize_memory:
                # prediction error: euclidean distance of the last model and the current scene vector
                if ii > 0:
                    # model = self.event_models[self.k_prev]
                    x_hat[ii, :] = x_hat_active
                    pe[ii] = np.linalg.norm(x_curr - x_hat_active)
                    # surprise[ii] = log_like[ii, self.k_prev]

            self.c[k] += self.kappa  # update counts
            # tan's code to not training while inferring
            if train:
                # update event model
                if not event_boundary:
                    # we're in the same event -> update using previous scene
                    assert self.x_prev is not None
                    self.event_models[k].update.remote(self.x_prev, x_curr)
                else:
                    # new event and not the only event, initialize by a world model
                    if k == len(active) - 1 and k != 0:
                        # increase n_epochs for new events
                        # self.event_models[k].n_epochs = int(self.event_models[k].n_epochs * 5)
                        self.event_models[k].set_n_epochs.remote(int(ray.get(self.event_models[k].get_n_epochs.remote()) * 5))

                        # set weights based on the general event model,
                        # always use .model_weights instead of .model.get_weights() or .model.set_weights(...)
                        # because .model is a common model used by all event models, its weights are of the last model being used
                        # self.event_models[k].model_weights = self.general_event_model.model_weights
                        self.event_models[k].set_model_weights.remote(self.general_event_model.model_weights)

                        # we're in a new event token -> update the initialization point only
                        self.event_models[k].new_token.remote()
                        # self.event_models[k].update_f0(x_curr)
                        if self.x_prev is None:  # start of each run
                            # assume that the previous scene is the same scene, so that not using update_f0
                            self.event_models[k].update.remote(x_curr, x_curr)
                        else:
                            self.event_models[k].update.remote(self.x_prev, x_curr)
                        # restore n_epochs
                        # self.event_models[k].n_epochs = int(self.event_models[k].n_epochs / 5)
                        self.event_models[k].set_n_epochs.remote(int(ray.get(self.event_models[k].get_n_epochs.remote()) / 5))

                        # update f0 for general model as well
                        # x_train_example = np.reshape(
                        #     unroll_data(self.general_event_model.filler_vector.reshape((1, self.d)), self.general_event_model.t)[-1, :, :],
                        #     (1, self.general_event_model.t, self.d)
                        # )
                        # self.general_event_model.training_pairs.append(tuple([x_train_example, x_curr.reshape((1, self.d))]))
                    else:
                        # we're in a new event token -> update the initialization point only
                        self.event_models[k].new_token.remote()
                        # self.event_models[k].update_f0(x_curr)
                        if self.x_prev is None:  # start of each run
                            # assume that the previous scene is the same scene, so that not using update_f0
                            self.event_models[k].update.remote(x_curr, x_curr)
                        else:
                            self.event_models[k].update.remote(self.x_prev, x_curr)

            self.x_prev = x_curr  # store the current scene for next trial
            if k == len(active) - 1 and not train:
                logger.warning(f'Creating a new event event with alfa={self.alfa} while doing validation, ignoring!!!')
            else:
                self.k_prev = k  # store the current event for the next trial

        # calculate Bayesian Surprise
        # tan's intepretation (might be wrong):
        # from t to t+1, the larger difference between posterior distribution at t and likelihood distribution at t+1, the larger surprise
        log_post = log_like[:-1, :] + log_prior[:-1, :]
        log_post -= np.tile(logsumexp(log_post, axis=1), (np.shape(log_post)[1], 1)).T
        surprise = np.concatenate([[0], logsumexp(log_post + log_like[1:, :], axis=1)])

        # Remove null columns to optimize memory storage, only for some arrays.
        self.results = Results()
        post = post[:, np.any(post != 0, axis=0)]
        self.results.post = post
        self.results.pe = pe
        self.results.surprise = surprise
        log_like = log_like[:, np.any(log_like != -np.inf, axis=0)]
        self.results.log_like = log_like
        log_prior = log_prior[:, np.any(log_prior != -np.inf, axis=0)]
        self.results.log_prior = log_prior
        # Should derive e_hat from post, avoid two-sources problem.
        # self.results.e_hat = np.argmax(log_like + log_prior, axis=1)
        self.results.e_hat = np.argmax(post, axis=1)
        self.results.x_hat = x_hat
        # self.results.log_loss = logsumexp(log_like + log_prior, axis=1)
        # self.results.log_boundary_probability = log_boundary_probability

        # self.results.restart_indices = restart_indices
        # self.results.repeat_indices = repeat_indices
        self.results.boundaries = boundaries
        # self.results.frame_dynamics = frame_dynamics
        self.results.c = self.c.copy()
        # self.results.Sigma = {i: self.event_models[i].Sigma for i in self.event_models.keys()}
        if minimize_memory:
            self.clear_event_models()
            return
        return post

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
            self.k += 1
            self._update_state(x, self.k)

            # pull the relevant items from the results
            if self.results is None:
                self.results = Results()
                post = np.zeros((1, self.k))
                log_like = np.zeros((1, self.k)) - np.inf
                log_prior = np.zeros((1, self.k)) - np.inf
                if save_x_hat:
                    x_hat = np.zeros((n_scene, self.d))
                    sigma = np.zeros((n_scene, self.d))
                    scene_log_like = np.zeros((n_scene, self.k)) - np.inf  # for debugging

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
                while k0 < self.k:
                    post = np.concatenate([post, np.zeros((n, 1))], axis=1)
                    log_like = np.concatenate([log_like, np.zeros((n, 1)) - np.inf], axis=1)
                    log_prior = np.concatenate([log_prior, np.zeros((n, 1)) - np.inf], axis=1)
                    n, k0 = np.shape(post)

                    if save_x_hat:
                        scene_log_like = np.concatenate([
                            scene_log_like, np.zeros((np.shape(scene_log_like)[0], 1)) - np.inf
                        ], axis=1)

                # extend the size of the posterior, etc
                post = np.concatenate([post, np.zeros((1, self.k))], axis=0)
                log_like = np.concatenate([log_like, np.zeros((1, self.k)) - np.inf], axis=0)
                log_prior = np.concatenate([log_prior, np.zeros((1, self.k)) - np.inf], axis=0)
                if save_x_hat:
                    x_hat = np.concatenate([x_hat, np.zeros((n_scene, self.d))], axis=0)
                    sigma = np.concatenate([sigma, np.zeros((n_scene, self.d))], axis=0)
                    scene_log_like = np.concatenate([scene_log_like, np.zeros((n_scene, self.k)) - np.inf], axis=0)

        else:
            log_like = np.zeros((1, self.k)) - np.inf
            log_prior = np.zeros((1, self.k)) - np.inf

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
