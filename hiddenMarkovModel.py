import numpy as np
import copy
import matplotlib.pyplot as plt


class HiddenMarkovModel:
    """First-order hidden Markov model with four inference methods: filtering, 
    prediction, smoothing and finding most likely explaination"""

    def __init__(self,priorDistribution,transitionMatrix,emissionMatrices,observations):
        self.priorDistribution = priorDistribution
        self.transitionMatrix = transitionMatrix
        self.emissionMatrices = emissionMatrices
        self.observations = observations


    def forward(self,t):
        """
        Forward inference algorithm used to calculate the forward message 
        when either doing filtering or smoothing operations
        """
        if t == 1: # Reached the last state with available evidence
            previousMessage = self.priorDistribution
        else:
            previousMessage = self.forward(t-1)

        observation = self.observations[t-1] 
        emissionMatrix = self.emissionMatrices[observation]
        unormalizedForwardMessage = np.dot(np.dot(np.transpose(self.transitionMatrix),previousMessage),emissionMatrix)
        
        # Normalization
        alpha = np.sum(unormalizedForwardMessage)
        forwardMessage = unormalizedForwardMessage/alpha

        return forwardMessage


    def backward(self,t,k):
        """
        Backward inference algorithm used to calculate the backward message
        when doing smoothing operations
        """
        # Reached the last state with available evidence
        if k == t: 
            previousMessage = np.ones(2)  
        else:
            previousMessage = self.backward(t,k+1)
    
        observation = self.observations[k-1]
        emissionMatrix = self.emissionMatrices[observation]
        backwardMessage = np.dot(np.dot(self.transitionMatrix,emissionMatrix),previousMessage)
        
        return backwardMessage


    def predict2(self,t,k):
        """
        Recursive subfunction used in the predict function
        """
        if k == 1: # Edge case where t=0, i.e. no evidence provided
            previousMessage = self.priorDistribution
        elif k-1 == t: # Switch to filtering when there's available evidence
            previousMessage = self.forward(t)
        else:
            previousMessage = self.predict2(t,k-1)

        predictionMessage = np.dot(np.transpose(self.transitionMatrix),previousMessage)
        return predictionMessage
    

    def getPathMessages(self,t,messages):
        """
        Recursive subfunction used in the argMax function
        """
        # The first message is the same as filtering X_1.
        if t == 1:
            m_11 = self.forward(1) 
            messages.append(m_11)
            return m_11
 
        observation = self.observations[t-1]
        emissionMatrix = self.emissionMatrices[observation]

        # Finding the maximum vector row-wise 
        maxVector = np.max(np.multiply(np.transpose(self.transitionMatrix),self.getPathMessages(t-1,messages)),axis=1)
        unormalizedVector = np.dot(emissionMatrix,maxVector)
        messages.append(unormalizedVector)

        return unormalizedVector
   

    def filter(self,t):
        """
        Recursive state estimation algorithm that utilizes the history of evidence
        """
        if t > len(self.observations):
            raise TypeError(f"Not enough observations to estimate state X_{t}")
        elif t <= 0:
            raise TypeError(f"Can only estimate states with t>=1.")

        return self.forward(t)
  

    def predict(self,t,k):
        """
        Recursive state prediction algorithm
        """
        if k <= t:
            raise TypeError(f"k has to be larger than t.")
        elif t < 0: 
            raise TypeError(f"Can only predict states with t>=0.")

        return self.predict2(t,k)


    def smooth(self,t):
        """
        Implementation of the Forward-Backward inference algorithm that compute the posterior
        marginals of all hidden state variables given a sequence of observations
        """
        if t > len(self.observations):
            raise TypeError(f"t cannot be larger than the number of observations.")
        elif t < 0:
            raise TypeError(f"Can only smooth states with t>=0.")
        
        smoothedEstimates = [0]*(t+1)
        forwardMessages = [0]*(t+1)
        forwardMessages[0] = self.priorDistribution
        for i in range(1,t+1):
            forwardMessages[i] = self.forward(i)
       
        for i in range(t,-1,-1):
            backwardMessage = self.backward(len(self.observations),i+1) # All observations should be used when smoothing
            unormalizedEstimate = np.multiply(forwardMessages[i],backwardMessage) # Hadamard/element-wise product
            alpha = np.sum(unormalizedEstimate)
            smoothedEstimates[i] = unormalizedEstimate/alpha 
            
        return smoothedEstimates


    def argMax(self,t):
        """
        Recursive inference algorithm that finds the most likely explaination 
        that state X_t is either True or False as a path of the previous states
        """

        if t > len(self.observations):
            raise TypeError(f"t cannot be larger than the number of observations.")
        elif t < 0:
            raise TypeError(f"Can only find most likely explaination for states with t>=0.")

        messages = []
        self.getPathMessages(t,messages)

        paths = {True:[True],False:[False]}
        for boolean in [True,False]:
            choice = boolean

            # Working backwards by calculating the best predecessor given that the current
            # state is either True or False
            for i in range(t-1,0,-1):
                index = 0 if choice else 1

                # The best predecessor is measured by the product of the distribution of the
                # previous state and the distribution of transitioning  to the current state
                currentDistribution = np.transpose(self.transitionMatrix)[index]
                previousMessage = messages[i-1]
                arrowDistribution = np.multiply(currentDistribution,previousMessage) 

                # Choosing the value for the current state that maximize the probability
                choice = np.argmax(arrowDistribution) 
                paths[boolean].insert(0,True if choice == 0 else False)

        return paths


def umbrellaworld():
    """
    You are the security guard stationed at a secret underground installation. You want to know whether it’s 
    raining today, but your only access to the outside world occurs each morning when you see the director 
    coming in with, or without, an umbrella. The following domain theory is proposed:
        1. The prior probability of rain is 0.5
        2. The probability of rain on day t is 0.7 given it rained on day t - 1 and 0.3 if not.
        3. The probability of the director coming in with an umbrella on day t if there's rain 
           on the same day is 0.9, and 0.2 if not.

    The following evidence is given:
        • e1 = {umbrella}
        • e2 = {umbrella}
        • e3 = {no umbrella}
        • e4 = {umbrella}
        • e5 = {umbrella}
    """
    priorDistribution = np.array([0.5,0.5])
    observations = [True,True,False,True,True]
    TransitionMatrix = np.array([[0.7,0.3],[0.3,0.7]])
    emissionMatrices = {
        True:np.array([[0.9,0],[0,0.2]]),
        False:np.array([[0.1,0],[0,0.8]])
    }
    return HiddenMarkovModel(priorDistribution,TransitionMatrix,emissionMatrices,observations)


def lakeworld():
    """
    Some tourists are curious if there are fish in a nearby lake. They are unable to observe whether this
    is true or not by staring into the lake. However, they can observe whether or not there are birds
    nearby that affect the presence of fish. Based on their instincts, the tourists propose the following
    domain theory:
        1. The prior probability of fish nearby (that is, without any observation) is 0.5.
        2. The probability of fish nearby on day t is 0.8 given there are fish nearby on day t − 1, and 0.3
           if not.
        3. The probability of birds nearby on day t if there are fish nearby on the same day is 0.75, and
           0.2 if not.
    
    The following evidence is given:
        • e1 = {birds nearby}
        • e2 = {birds nearby}
        • e3 = {no birds nearby}
        • e4 = {birds nearby}
        • e5 = {no birds nearby}
        • e6 = {birds nearby}
    """
    priorDistribution = np.array([0.5,0.5])
    TransitionMatrix = np.array([[0.8,0.2],[0.3,0.7]])
    observations = [True,True,False,True,False,True]
    emissionMatrices = {
        True:np.array([[0.75,0],[0,0.2]]),
        False:np.array([[0.25,0],[0,0.8]])
    }
    return HiddenMarkovModel(priorDistribution,TransitionMatrix,emissionMatrices,observations)
   

if __name__ == '__main__':
    #u = umbrellaworld()
    h = lakeworld()
    
    print(f"--------Task 1b : filtering--------")
    results = []
    for i in range(1,7):
        result = h.filter(i)
        print(f"P(X_{i}|e_1:{i}) = <{round(result[0],3)},{round(result[1],3)}>")
        results.append(result[0])
    print("\n")
    plt.plot(range(1,7),results)
    plt.ylabel('p')
    #plt.show()

    print(f"--------Task 1c : predicting--------")
    results = []
    for i in range(7,31):
        result = h.predict(6,i) # t=6
        print(f"P(X_{i}|e_1:6) = <{round(result[0],5)},{round(result[1],5)}>")
        results.append(result[0])
    print("\n")
    plt.plot(range(7,31),results)
    plt.ylabel('p')
    #plt.show()

    print(f"--------Task 1d : smoothing--------")
    results = h.smooth(5)
    for i in range(6):
        print(f"P(X_{i}|e_1:6) = <{round(results[i][0],3)},{round(results[i][1],3)}>")
    print("\n")
    plt.plot(range(1,7),results)
    plt.ylabel('p')
    #plt.show()
    
    print(f"--------Task 1d : most likely explaination--------")
    for i in range(1,7):
        result = h.argMax(i)
        xOutput = "x_1:"+str(i)
        for boolean in [True,False]:
            print(f"arg max {xOutput} P({xOutput} x_{i} == {boolean}|e_1:{i}) = {result[boolean]}")
 
    
   
    
   

   

    

    
    