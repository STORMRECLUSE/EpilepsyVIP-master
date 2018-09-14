'''
This file is structured similarly to scikit-learn's perceptron code, currently in beta. Similar structures.
'''
from __future__ import print_function
from __future__ import division
import numpy as np
import six, warnings
from sklearn.utils.fixes import signature

def lin_func(data,x1,y1,m):
    if type(x1) is float:
        x1 = np.array([x1])
    return np.asarray(m)[:,None]*(data-np.asarray(x1)[:,None]) + y1

def scale_range(data,min_in,max_in,min_out,max_out):
    return lin_func(data,min_in,min_out,float(max_out-min_out)/(max_in-min_in))

class BaseMLP(object):
    '''
    Makes a multilayer perceptron class.
    '''
    def __init__(self,hidden_layers = (100,),object_func = 'squares',trans_func = 'tanh',
                 slope_param = 1.,learn_rate = .001,momentum = 0.6, weight_range = .1, normalizer = 'variance',
                 output_range_percent = .85, batchsize = 1,max_steps = 100000, err_conv = 0.05):
        def check_inputs(object_func,trans_func,momentum,normalizer):
            #check error function
            if object_func =='squares':
                self.ofunc = self.square_err
                self.ofunc_grad = self.square_err_grad
            else:
                raise ValueError('That objective function is not available.')
            #check transfer function
            if trans_func =='tanh':

                self.transf_str = trans_func
                self.tfunc = self.tanh_fxn
                self.tfunc_dx = self.tanh_deriv

                self.min_out = -output_range_percent
                self.max_out = output_range_percent
            elif trans_func =='sigmoid':
                self.transf_str = trans_func
                self.tfunc = self.sigmoid_fxn
                self.tfunc_dx = self.sigmoid_der

                self.min_out = (1.-output_range_percent)/2.
                self.max_out = (1+output_range_percent)/2.
            else:
                raise ValueError('That transfer function is not available.')

            #check momentum
            if 0<=momentum<=1:
                self.m = momentum
            else:
                raise ValueError('That momentum was valid. Only values between 0 and 1.')

            #check normalizer
            if normalizer == 'variance':
                self.norm_func = self._variance_normalizer
            elif normalizer == 'range':
                self.norm_func = self._range_normalizer
            else:
                raise ValueError('That normalizing function is not available.')

        check_inputs(object_func,trans_func,momentum,normalizer)
        self.transf_slope = slope_param
        self.l = learn_rate
        self.weight_range = weight_range
        self.hidden_lens = hidden_layers
        self.batch = batchsize
        self.max_steps = max_steps
        self.err_conv = err_conv

    @classmethod
    def _get_param_names(cls):
        """ (Lifted from sklearn base) Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])


    def get_params(self, deep=True):
        """Get parameters for this estimator. (Lifted from sklearn)
        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """ (Lifted from the beta version of the sklearn estimator) Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The former have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in six.iteritems(params):
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (key, self.__class__.__name__))
                setattr(self, key, value)
        return self

    def fit(self,inputs,outputs,learn_hist = False,learn_step = 4, print_progress = False ):
        '''
        Train the neural network on the inputs and outputs. Scales everything for you.
        Will output a learning history if that is set to true
        :param inputs: A k x n matrix, where k is the number of observations, and n is the number of features
        :param outputs: A k x p matrix, where k is as above and p is the number of outputs
        :param learn_hist: bool
        :return:
        '''
        self.weights,self.biases = self._initialize_weights(
            np.size(inputs,axis=1),self.hidden_lens,np.size(outputs,axis=1),self.weight_range)
        scaled_inputs,scaled_outputs= self.data_normalizer(inputs.T,outputs.T)
        change_weights = [np.zeros_like(weight) for weight in self.weights]
        change_bias = [np.zeros_like(bias) for bias in self.biases]
        #deltas = [np.zeros((np.size(inputs,axis=0),input_len)) for input_len in (np.size(inputs,axis=1),) + self.hidden_lens]
        history = []
        kwargs = {}
        for i in range(self.max_steps):
            indices_chosen = np.random.choice(np.size(scaled_inputs,1),size = self.batch, replace=False)
            input_chosen, desired_chosen = scaled_inputs[:,indices_chosen], scaled_outputs[:,indices_chosen]
            layers_out, derr= self._evaluate(input_chosen,d_err=True,desired= desired_chosen,**kwargs)
            err = self._error(self._evaluate(scaled_inputs)[-1],scaled_outputs)
            if i==0:
                kwargs = {'old_layers': layers_out}
            if not i % learn_step:
                if print_progress:
                    print('Step {}, Error: {}'.format(i,err))
                history.append(err)
            if np.abs(err) < self.err_conv:
                if learn_hist:
                    return np.array(history)
                else:
                    return
            change_weights,change_bias = \
                self.update_weights(layers_out,derr,prev_change_weights=change_weights,prev_change_bias=change_bias
                                     )
        print('Network did not converge in time!')
        return np.array(history)


    def _error(self, inputs, desired):
        return self.ofunc(inputs,desired)

    def _evaluate(self,inputs, d_err = False, desired= None, old_layers = None):

        layer_out = inputs
        if old_layers is not None:
            layers_out = old_layers
        else:
            layers_out = [layer_out] + [0]*len(self.weights)
        for ind,(weight_mat,bias_vect) in enumerate(zip(self.weights,self.biases)):
            layer_out = self.tfunc(np.dot(weight_mat,layer_out) + bias_vect[:,None], self.transf_slope)
            layers_out[ind + 1] = layer_out

        return_val = layers_out
        if d_err and desired is not None:
            return_val = (return_val, self.ofunc_grad(layer_out,desired))
        return return_val


    def predict(self, unscaled_inputs):
        scaled_inputs = lin_func(unscaled_inputs.T,self.inputx,self.inputy,self.inputm)

        scaled_outputs = self._evaluate(scaled_inputs)[-1]
        unscaled_output = lin_func(scaled_outputs,self.outputy,self.outputx,1/self.outputm)
        return unscaled_output

    def point_error(self,unscaled_inputs,unscaled_desired):
        scaled_inputs =  lin_func(unscaled_inputs.T,self.inputx,self.inputy,self.inputm)
        scaled_outputs = self._evaluate(scaled_inputs)[-1]
        scaled_desired = lin_func(unscaled_desired.T,self.outputx,self.outputy,self.outputm)
        return self.ofunc(scaled_outputs,scaled_desired)

    def update_weights(self,layers_out,derr, prev_change_weights, prev_change_bias):
        change_weights,change_bias = prev_change_weights,prev_change_bias
        #deltas = prev_deltas
        delta = -derr*self.tfunc_dx(layers_out[-1], self.transf_slope)
        change_weights[-1] = self.l*np.dot(delta,layers_out[-2].T) + self.m*prev_change_weights[-1]
        change_bias[-1] = self.l *np.sum(delta,axis=1)+ self.m*prev_change_bias[-1]
        #change_weights = [self.l* np.outer(layers_out[-1],deltas) + self.m*prev_change_weights[-1]]
        for ind,(layer_out,weights) in enumerate(zip(layers_out[-2:0:-1],self.weights[-1::-1])):
            delta = np.dot(weights.T,delta)*self.tfunc_dx(layer_out,self.transf_slope)
            #deltas[ind] = delta
            #change_weights.append(self.l*np.outer(layer_out,deltas[-1] + self.m*prev_change_weights)
            change_weights[-ind-2] = self.l*np.dot(delta,layers_out[-ind-3].T) + self.m*prev_change_weights[-ind-2]
                          #for layer_out,prev_change, delta in zip(layers_out,prev_change_weights,deltas[-1::-1])]

            change_bias[-ind-2] = self.l *np.sum(delta,axis=1)+ self.m*prev_change_bias[-ind-2]
                          #for prev_change, delta in zip(prev_change_bias,deltas[-1::-1])]
        for weight, bias, weight_change, bias_change in zip(self.weights,self.biases,change_weights,change_bias):
            weight +=weight_change
            bias += bias_change
        return change_weights,change_bias #deltas


    @staticmethod
    def _initialize_weights(n_inputs,n_hidden,n_outputs, weight_range):
        weights, biases = [],[]
        total_list = n_hidden + (n_outputs,)
        last_layer = n_inputs
        for this_layer in total_list:
            weights.append((np.random.rand(this_layer,last_layer)-.5)*2*weight_range)
            biases.append((np.random.rand(this_layer)-.5)*2*weight_range)
            last_layer = this_layer
        return weights,biases

    def data_normalizer(self,inputs,outputs):
        self.norm_func(inputs)

        min_rawout, max_rawout = np.min(outputs, axis=1), np.max(outputs,axis=1)
        self.outputx = min_rawout
        self.outputy = self.min_out
        self.outputm = (self.max_out-self.min_out)/(max_rawout-min_rawout)

        scaled_inputs =lin_func(inputs,self.inputx,self.inputy,self.inputm)
        scaled_outputs = lin_func(outputs,self.outputx,self.outputy,self.outputm)
        return scaled_inputs,scaled_outputs

    def _range_normalizer(self,inputs):
        min_rawinput, max_rawinput = np.min(inputs, axis=1), np.max(inputs,axis=1)

        self.inputx = min_rawinput
        self.inputy = -1.5
        self.inputm = 3/(max_rawinput-min_rawinput)


    def _variance_normalizer(self,inputs):
        self.inputx = np.mean(inputs,axis=1)
        self.inputy = 0.
        self.inputm = 1/np.std(inputs,axis=1)



    @staticmethod
    def square_err(calc,des):
        diff = np.atleast_2d(calc - des)
        return np.mean(np.sqrt(np.einsum('ij,ij ->j',diff,diff)/np.size(diff,axis=0)))

    @staticmethod
    def square_err_grad(calc,des):
        return (calc-des)

    @staticmethod
    def sigmoid_fxn(x,slope):
        return 1./(1 + np.exp(-slope*x))

    @staticmethod
    def sigmoid_der(sig,slope):
        '''
        Note: sig is the OUTPUT of the sigmoid fxn!
        :param sig:
        :param slope:
        :return:
        '''
        return slope*sig*(1-sig)

    @staticmethod
    def tanh_fxn(x,slope):
        return np.tanh(slope*x)

    @staticmethod
    def tanh_deriv(tanh,slope):
        '''
        Note: tanh is the OUTPUT of tanh!
        :param tanh:
        :param slope:
        :return:
        '''
        return slope*(1-tanh**2)

class MLPClassifier(BaseMLP):
    #TODO: change external representation of classifying data, using softmax, etc
    pass


class AutoEncoder(BaseMLP):

    def __init__(self,hidden_layers = (100,),object_func = 'squares',trans_func = 'tanh',
                 slope_param = 1.,learn_rate = .001,momentum = 0.6, weight_range = .1, normalizer = 'variance',
                 output_range_percent = .85, batchsize = 1,max_steps = 100000, err_conv = 0.03, outlier_frac = 0.05):
        if not 0<=outlier_frac<=1:
            raise ValueError('Outlier fraction must be between 0 and 1 inclusive')
        self.outlier_frac = outlier_frac

        super(AutoEncoder,self).__init__(hidden_layers= hidden_layers,object_func = object_func,trans_func = trans_func,
                slope_param = slope_param,learn_rate = learn_rate,momentum = momentum, weight_range = weight_range,
                normalizer = normalizer,output_range_percent = output_range_percent, batchsize = batchsize,
                max_steps = max_steps, err_conv = err_conv)

    def fit(self,inputs, learn_hist = False,print_progress = True,learn_step = 4 ):
        return_res = super(AutoEncoder,self).fit(inputs,inputs,learn_hist=learn_hist,
                                                 print_progress = print_progress,learn_step=learn_step)

        self._store_percentile(inputs)

        return return_res

    def _store_percentile(self,inputs):
        '''
        Set the threshold beyond which
        :param inputs:
        :return:
        '''
        scaled_outputs = lin_func(inputs.T,self.outputx,self.outputy,self.outputm)
        scaled_inputs = lin_func(inputs.T,self.inputx,self.inputy,self.inputm)
        diff = self._evaluate(scaled_inputs)[-1]- scaled_outputs
        self.errs = np.apply_along_axis(lambda x: self.point_error(x,x),0,diff)
        return

    def percentile_estimator(self,inputs):
        '''
        Return the percentile of the error that your inputs are.
        :param inputs:
        :return:
        '''
        #First, scale the outputs of the estimator
        scaled_outputs = lin_func(inputs.T,self.outputx,self.outputy,self.outputm)
        scaled_inputs = lin_func(inputs.T,self.inputx,self.inputy,self.inputm)
        diff = self._evaluate(scaled_inputs)[-1]- scaled_outputs
        err = np.apply_along_axis(lambda x: self.point_error(x,x),0,diff)
        percentiles = np.searchsorted(self.errs,err)/len(self.errs)
        return percentiles

    def predict(self,inputs):
        percentiles = self.percentile_estimator(inputs)
        return (1-percentiles) < self.outlier_frac

