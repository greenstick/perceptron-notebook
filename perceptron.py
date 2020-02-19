#!/usr/bin/python3
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera

class Perceptron(object):
    """
    An Example of Rosenblatt's Perceptron for Binary Classification
    """
    def __init__(self,
        eta                         = 10e-3,
        epochs                      = 1,
        batch_size                  = 1,
        activation_function         = "signum",
        error_function              = "rosenblatt_error",
        weight_function             = "random_weights"
    ):
        # Set Attributes With Defaults
        self.eta                    = eta
        self.epochs                 = epochs
        self.batch_size             = batch_size
        self.activation_function    = activation_function
        self.error_function         = error_function
        self.weight_function        = weight_function
        
        # Other References (Initalized During Training)
        self.classes                = None
        self.features               = None
        self.errors                 = None
        self.outputs                = None
        self.weights                = None
        self.history                = None
    
    #
    # Activation Function
    #
    
    # Signum Activation
    def signum (self, product):
        return np.where(product >= 0, 1.0, -1.0)
    
    #
    # Weight Initializer
    #
    
    # Random Weights Initializer
    def random_weights(self, features):
        weights = np.random.normal(0, 1, (features.shape[1] + 1, 1)) # We Add 1 Here For Our Bias Term
        return weights
    
    #
    # Error Function
    #
    
    # Rosenblatt's Error
    def rosenblatt_error (self, classes, features):
        misclassified_classes = np.where(classes != self.outputs, classes - self.outputs, 0)
        error = np.dot(features.T, misclassified_classes)
        return error
    
    #
    # Model Helper Methods
    #
    
    # Inserts a Bias Term (Intercept) Into Our Features
    def insert_bias (self, features):
        n = features.shape[0]
        bias = np.ones(n).reshape(n, 1)
        return np.concatenate((bias, features), axis = 1)
    
    # Classifies Our Example(s)
    def classify (self, features, weights):
        return getattr(self, self.activation_function)(np.dot(features, weights))
        
    #
    # Model Methods
    #
        
    # Our Training Function
    def train(self,
        classes,
        features,
        weights                     = None,
        eta                         = None,
        epochs                      = None,
        batch_size                  = None,
        activation_function         = None,
        error_function              = None,
        weight_function             = None
    ):
        # Assign Our Classes & Features to The Perceptron
        self.classes                = classes[:, np.newaxis] # This Structures Our Classes Into a 2D Numpy.array
        self.features               = self.insert_bias(features)  # This Inserts a Bias Term (Intercept) Into Our Features
        # If We've Received Values For These, Assign Them, Otherwise Use Default Init Values
        self.eta                    = eta                   if eta                  is not None else self.eta
        self.epochs                 = epochs                if epochs               is not None else self.epochs
        self.batch_size             = batch_size            if batch_size           is not None else self.batch_size
        self.activation_function    = activation_function   if activation_function  is not None else self.activation_function
        self.error_function         = error_function        if error_function       is not None else self.error_function
        self.weight_function        = weight_function       if weight_function      is not None else self.weight_function
        # Init Training History
        self.history = []
        
        # Let's Generate Our Weights
        if weights is None:
            weights = getattr(self, self.weight_function)(features)
            self.history.append(weights)
            self.weights = weights
        
        # Epoch Loop
        for i in range(self.epochs):
            
            # Sample Loop
            for j in range((self.features.shape[0] // self.batch_size) + 1):
                
                # Generate Random Index & Grab A Random Minibatch
                indices = np.random.permutation(self.features.shape[0])
                classes = self.classes[indices[:, np.newaxis], np.arange(self.classes.shape[1])][0:self.batch_size]
                features = self.features[indices[:, np.newaxis], np.arange(self.features.shape[1])][0:self.batch_size]
                
                # Now Let's Call Our Classification Function & Compute Our Error
                self.outputs = self.classify(features, self.weights)
                self.errors = getattr(self, self.error_function)(classes, features)
                
                # Update Our Weights
                weights_delta = self.eta * self.errors
                weights_delta_total = np.sum(weights_delta, axis = 1)[:, np.newaxis] # Sum The Updates Across Entire Batch
                self.weights = self.weights + weights_delta_total

                # Update Our Training History
                self.history.append(self.weights)
                
        return self
    
    # Our Prediction Function (For Both Validation & Prediction)
    def predict (self, features):
        features = self.insert_bias(features)  # This Inserts a Bias Term (Intercept) Into Our Features
        return self.classify(features, self.weights)
    
    #
    # Visualization Methods
    #
    
    # Method For Visualizing Training - Relies on Celluloid Library For Animation
    def visualize_training (self, 
        df, 
        label,
        features, 
        history = [],
        resolution = 0.01, 
        interval = 2, 
        blit = True, 
        loop = False
    ):
        
        # Use History Attribute if Not History Is Provided
        history = history if len(history) > 0 else self.history
        # Get Classes & Features, Insert Bias Term
        classes, features = df[[label]].values[:,0], df[features].values

        # Setup Plotting
        plt.figure(figsize = (12, 12))
        fig, ax = plt.subplots()
        camera = Camera(fig)  

        # Show Animation Rendering Progress
        self.print_progress_bar(0, len(history), prefix = "Rendering:", suffix = "Complete", length = 50)  
        
        # Replay Training
        for i, weights in enumerate(history):

            # Visualize Decision Boundary
            x1, x2 = self.get_decision_boundary()
            ax.plot(x1, x2)
            # Plot Data Points
            ax.scatter(features[:,0], features[:,1], c = classes, alpha = 0.6, cmap = plt.cm.Dark2)
            
            # Save Animation Frame
            camera.snap()

            # Update Animation Rendering Progress
            self.print_progress_bar(i + 1, len(history), prefix = "Rendering:", suffix = "Complete", length = 50)  
        
        # Generate Animation
        animation = camera.animate(interval = interval, blit = blit, repeat = loop)

        return animation
    
    # Method For Visualizing Decision Boundary of Trained Perceptron
    def visualize_decision_boundary (self, 
        df, 
        label, 
        features, 
        resolution = 0.01
    ):
        
        # Get Classes & Features, Insert Bias Term
        classes, features = df[[label]].values[:,0], df[features].values

        # Setup Plotting
        fig, ax = plt.subplots(figsize = (12, 12))   
        # Visualize Decision Boundary
        x1, x2 = self.get_decision_boundary()
        ax.plot(x1, x2)
        # Plot Data Points
        ax.scatter(features[:,0], features[:,1], c = classes, cmap = plt.cm.Dark2)
        
        return fig, ax
    
    #
    # Visualization Helper Methods
    #

    # Extract out the decision boundary for the trained model
    def get_decision_boundary (self):
        w0, w1, w2 = self.weights
        xDomainMax = np.max(features[:, 0])
        decision_boundary = (np.linspace(0, xDomainMax, xDomainMax + 1), (np.linspace(0, xDomainMax, xDomainMax + 1) * (w1 / -w2)) + (w0 / -w2))
        return decision_boundary

    # Print iterations progress
    def print_progress_bar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()
            
if __name__ = "__main__":
    pass
else:
    pass
