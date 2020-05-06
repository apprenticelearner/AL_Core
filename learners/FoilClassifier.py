import sys
import math
import string
import numpy as np
import copy
from sklearn import datasets
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
import random 

# References:
# https://cgi.csc.liv.ac.uk/~frans/KDD/Software/FOIL_PRM_CPAR/foil.html
# https://en.wikipedia.org/wiki/First-order_inductive_learner
# http://andrefreitas.org/symbolic_ai/lecture_16.pdf



class param:
    MIN_BEST_GAIN = 0.1 # Minimum gain threshold 
    MAX_NUM_IN_ANT = 2  # Maximum number of antecedents for rules
    K = 3               # Number of rules to consider when calculating average accuracy 

# --------------------------------UTILITIES--------------------------------


# Convert a list of dictionary to a transformed 2D numpy array, each cell annotated
# with the feature name. In training, the dictVect initialized can be stored in a FOIL
# object. When predicting, the test data can be transformed to a 2D numpy array using
# the dictVect.
def convertX(dictList, Xtype, dv=None):
    if Xtype == "train":
        dv = DictVectorizer(sparse=False)
        X = dv.fit_transform(dictList)
    elif Xtype == "pred":
        X = dv.transform(dictList)
    nrow, ncol = X.shape[0], X.shape[1]
    nameArray = dv.get_feature_names()
    prefix = np.empty((nrow, ncol), dtype = 'object')
    for c in range(ncol):
        prefix[:,c] = nameArray[c] + "_"
    strData = X.astype(str)
    newX = np.add(prefix, strData)
    return (newX, dv)



def predAccu(tru, pred):
    match = np.sum(tru == pred) 
    total = len(tru)
    accu = (float(match) / float(total))
    return accu


# Sort list of tuples decreasingly according to the value of 2nd element in tuples
def sortTup(tup):
    tupSorted = sorted(tup, key=lambda x: x[1], reverse=True)
    return tupSorted



# Add prefix of feature name to cell values
def addFeature(data):
    nrow, ncol = data.shape[0], data.shape[1]
    prefix = np.empty((nrow, ncol), dtype = 'object')
    for c in range(ncol):
        prefix[:,c] = "f" + str(c) + "_"
    strData = data.astype(str)
    new = np.add(prefix, strData)
    return new


# Get positive and negative examples according to the specified label
# All rows with label not equal to the specified label are negative examples
def getPosOrNegExamples(feature, lab, pos):
    posInd, negInd = np.where(lab == pos)[0], np.where(lab != pos)[0]
    posEx, negEx = feature[posInd, :], feature[negInd, :]
    return [posEx, negEx]



# a is a 1D numpy array, s is a set
# Returns true only if all items in s are in a
def checkSubset(a, s):
    if not s: return True
    for i in s:
        if i not in a: return False
    return True


# Rank rules in decreasing accuracy 
def rankPair(rules, accu):  
    rankedR = [r for _, r in sorted(zip(accu, rules), reverse = True)]
    rankedA = sorted(accu, reverse = True)
    return [rankedR, rankedA]


def uniqueAttributes(a):
    return np.unique(a)



# All examples which do not contain attribute are removed
def retainExamples(attr, ex):
    if len(ex) == 0: return np.array([])
    # Apply checkSubset for each row in examples
    retain = np.apply_along_axis(checkSubset, 1, ex, attr)
    retainEx = ex[retain, :]
    return retainEx


# Remove all examples that satisfy the attribute
def removeExamples(attr, ex):
    assert(len(ex)!=0)
    # Apply checkSubset for each row in examples
    remove = np.invert(np.apply_along_axis(checkSubset, 1, ex, attr))
    removeEx = ex[remove, :]
    return removeEx



# Calculate gain for (P, P', N, N')
def gain(p, p2, n, n2):
    sP, sP2 = np.shape(p)[0], np.shape(p2)[0]
    sN, sN2 = np.shape(n)[0], np.shape(n2)[0]

    # Gain(a) = |P'| (log2(|P'|/|P'|+|N'|) - log2(|P|/|P|+|N|))
    if sP2 == 0: return 0 
    gain = sP2 * (math.log2(float(sP2) / float(sP2 + sN2)) - math.log2(float(sP) / float(sP + sN))) 
    return gain




def calculateGain(ant, pos, neg): # pos or neg could be empty
    assert(np.shape(pos)[1] != 0)
    # pos2 (P') is a subset of rows in pos (P) that contain the antecedents
    pos2 = retainExamples(ant, pos) 
    # neg2 (N') is a subset of rows in neg (N) that contain the antecedents
    neg2 = retainExamples(ant, neg)  
    g = gain(pos, pos2, neg, neg2) 
    return g







class FOILClassifier(object):
    """
    A class used to represent a FOIL Classifier
    
    Note:
    P: positive examples
    N: negative examples
    P': subset of P
    N': subset of N

    A rule ({antedecent1, antecedent2 ... }, label) is represented as a tuple,
    with the antecedent set as the if-clause, and label as the then-clause.

    One antecedent is expressed as a fact in the form "feature_value", 
    meaning feature being value is true.  
    eg. Antecedent "feature1_0" represents "feature1 equals 0"

    Rules are conjunctive, and antecedents are disjunctive.
    eg. ({A, B}, 1), ({C}, 0) represent (A and B -> 1) or (C -> 0)

    Methods
    -------
    startFOIL()
        Outer Loop of FOIL.
        Generate rules for classification by learning rules for all label classes.

    foilGeneration(rule, con)
        Inner Loop of FOIL. 
        Generate a rule by recursively adding new antecedents to it by choosing the 
        best gain with modfying P' and N'. 
        
    fit(feature, label, dv)
        Create a FOIL instance and fit with training data and a dictVectorizer

    predictLabels(newdata, k)
        Predict a label for one test data by picking the label whose associated rules
        have higher average accuracy.

    predictAll(newdata, k)
        Predict labels for all test data. 

    setPosNeg(feature, lab, posVal)
        Define which class label is positive, and get positive and negative examples 
        for that label.    

    reset()
        In outer loop of FOIL, reset P' and N' to P and N, and create an empty 
        antecedent set to learn a new rule.

    calculateGains()
        Attempt to add one new antecedent to learned antecedents for a rule, 
        calculate gain for the union of learned antecedents with one of all possible 
        unexplored antecedents to add to a disjunctive rule.

    noGain()
        Detect if any antecedent exists that can produce a gain above minimum.
    
    getAccuracy()
        Calculate accuracy for each rule generated.

    """

    def __init__(self, feature = None, label = None, dictVect = None):
        """
        Parameters
        ----------
        feature: 2D numpy array 
            Fit transformed training data from dictVect, with each cell as a
            possible antecedent
        label: 1D numpy array
            Labels for the training data
        dictVect: numpy dictVect 
            The dictVect for the original dict list training data
        """
        self.feature = np.array(feature)
        self.label = np.array(label)
        self.uniqueAttr = np.unique(feature)
        self.uniqueLab = np.unique(label)
        self.numAttribute = self.uniqueAttr.size
        self.numClass = np.unique(self.label).size

        # Keep track of an attribute array, with first row as an indicator row that
        # records whether an antecedent has been considered for inclusion in the 
        # if-clause, and second row that stores the gain for that antecedent
        self.attrArray = np.zeros((2,self.numAttribute)) 
        self.rules = [] 
        self.con = None # consequence, label, or then-clause for a rule 
        self.ruleAccu = []
        self.dictVect = dictVect
        

        # Those get changed once per outerloop
        self.posExamples = None
        self.negExamples = None


        # Those get changed inside the innerloop
        self.posExamples2 = None
        self.negExamples2 = None
        self.attrArray2 = None
        self.ant = set() # set of antecedents for a rule
        

    # Define the label to learn toward, so it will be set to then-clause later
    def setPosNeg(self, feature, lab, posVal):
        """
        Parameters
        ----------
        feature: 2D numpy data array 
        lab: 1D numpy labrel array
        posVal: integer
            The label for which FOIL learns toward, rows of data with this label
            will be set to positive examples 
        """
        [pos, neg] = getPosOrNegExamples(feature, lab, posVal)
        self.posExamples, self.negExamples = pos, neg
        self.con = posVal


    # Make copies of N, P and A, set antecedents to an empty set for inner loop,
    # in preparation for new rule generation
    def reset(self):
        self.ant = set()
        self.posExamples2 = copy.deepcopy(self.posExamples)
        self.negExamples2 = copy.deepcopy(self.negExamples)
        self.attrArray2 = copy.deepcopy(self.attrArray)


    # Calculate gain for each possible antecedent to add for a rule
    def calculateGains(self):

        attrCopy = copy.deepcopy(self.attrArray2)
        calculated = attrCopy[0]
        attrCopy[1] = np.zeros(len(attrCopy[1]))

        for i in range(len(calculated)):
            if not calculated[i]: # If that antecedent has not been considered before
                tempAnt = copy.deepcopy(self.ant)
                tempAnt.add(self.uniqueAttr[i])
                # the gain if new antecedent were to be added 
                attrCopy[1][i] = calculateGain(tempAnt, self.posExamples2, self.negExamples2) 

        return attrCopy


    # If adding any one of antecedents does not produce gain above minimum, return true 
    def noGain(self):
        attrGains = self.calculateGains()[1] 
        if np.any(attrGains > param.MIN_BEST_GAIN): return False
        return True


    # Calculate accuracy = (Nc+1)/(Ntot+numberOfClasses) for a rule, where
    # Nc = number of training data rows whose features contain the antecedents of the rule
    # and whose label matches the consequence of that rule
    # Ntot = number of training data rows whose features contain the antecedents of the rule
    # numberOfClasses: number of unique label classes
    def getAccuracy(self):
        ind = np.apply_along_axis(checkSubset, 1, self.feature, self.ant)
        total = np.sum(ind)
        lab = self.label[ind]
        count = np.sum(lab == self.con)
        accu = (float(count) + 1) / (float(total) + float(self.numClass))
        return accu




    # Keep looking for rules until N' is empty, or minimum gain threshold is reached, or 
    # maximum number of antecedents is reached 
    def foilGeneration(self, rule, con):
        """
        Parameters
        ----------
        rule: 1D list
            This rule list gets modified in each call of foilGeneration as more 
            antecedents are added
        con: integer
            The then-clause, the label for which FOIL is learning toward 
        """      
        self.attrArray2 = self.calculateGains()

        # index of the antecedent, adding which produces the maximum gain
        # np.argmax returns the index of the FIRST occurrence of the maximum value if there 
        # are multiple indices at which maximum appears
        maxInd = np.argmax(self.attrArray2[1]) 

        bestGain = self.attrArray2[1][maxInd]
        bestAttr = self.uniqueAttr[maxInd]

        # If adding more antecedents does not produce much gain, add the rule so far and return
        if bestGain <= param.MIN_BEST_GAIN: 
            rule.append((self.ant, con))
            return 

        # Else, Add the new disjunctive antecedent 
        self.ant.add(bestAttr) 
        # this antecedent has been considered for inclusion in the rule, set the indicator to 1
        self.attrArray2[0][maxInd] = 1
        
            
        # Update P' and N'
        # Remove from P' and N' examples that do not satisfy the if-clause
        self.posExamples2 = retainExamples(self.ant, self.posExamples2) 
        self.negExamples2 = retainExamples(self.ant, self.negExamples2)
    
        # If N' is empty, or the maximum number of antecedents added exceeds the limit, 
        # Add the rule so far and return
        if len(self.negExamples2) == 0 or len(self.ant) >= param.MAX_NUM_IN_ANT:

            rule.append((self.ant, con))
            return 

        self.foilGeneration(rule, con)





    # Keep looking for rules until P is empty
    def startFOIL(self):
        # Learn rules for each label class
        for labVal in self.uniqueLab:
            # Get P, and N according to the label
            self.setPosNeg(self.feature, self.label, labVal)        
            
            while len(self.posExamples) != 0:
                self.reset()

                # if no attributes exist that can produce a gain above minimum, break
                if self.noGain(): 
                    break


                newRule = [] # Initialize a new rule list
                self.foilGeneration(newRule, self.con) # new rule list has been modified 
                newAnt = newRule[0][0]
                accu = self.getAccuracy() # Calculate the accuracy for the new rule found


                self.rules.append(newRule[0])
                self.ruleAccu.append(accu)
                # Remove from P all examples which satisfy the new rule
                self.posExamples = removeExamples(newAnt, self.posExamples)   
                


    def getRulesAccu(self):
        rules, accu = [], []
        for r in range(len(self.rules)):
            rules.append(self.rules[r])
            accu.append(self.ruleAccu[r])

        [rankedRules, rankedAccu] = rankPair(rules, accu)
        return [rankedRules, rankedAccu]
    
            

    def fit(self, feature, label, dv):
        """
        Parameters
        ----------
        feature: 2D numpy array 
            Fit transformed training data from dictVect, with each cell as a
            possible antecedent
        label: 1D numpy array
            Labels for the training data
        dictVect: numpy dictVect 
            The dictVect for the original dict list training data
        """
        self.feature = np.array(feature)
        self.label = np.array(label)
        self.dictVect = dv
        self.__init__(feature, label, dv)
        self.startFOIL()

        



    def predictLabels(self, newdata, k):
        """
        Parameters
        ----------
        newdata: 2D numpy array 
            Transformed test data using self.dictVect
        k: integer
            The number of rules used to calculate average accuracy 
        """
        # Obtain all rules whose antecedent is a subset of the given test data
        candidateRules, candidateAccu = [], []
        for r in range(len(self.rules)):
            if checkSubset(newdata, self.rules[r][0]):
                candidateRules.append(self.rules[r])
                candidateAccu.append(self.ruleAccu[r])

        # Select the best K rules for each class according to their Laplace accuarcy
        res = []
        for labVal in self.uniqueLab:
            matchRule, matchAccu = [], []
            for i in range(len(candidateRules)):
                if candidateRules[i][1] == labVal: 
                    matchRule.append(candidateRules[i])
                    matchAccu.append(candidateAccu[i])
   

            [rankedRules, rankedAccu] = rankPair(matchRule, matchAccu)
            # Calculate average accuracy
            meanAccu = 0 if not rankedRules else sum(rankedAccu[:k]) / len(rankedAccu[:k])
            res.append((labVal, meanAccu))

        sortedTup = sortTup(res)
        # Return the label for which the rules have the highest average accuracy
        # If multiple labels have the same highest accuracy, return the first one in the list
        predLabel = sortedTup[0][0] 
        return predLabel



    def predictAll(self, newdata, k):
        """
        Parameters
        ----------
        newdata: 2D numpy array 
            Transformed test data using self.dictVect
        k: integer
            The number of rules used to calculate average accuracy 
        """
        # Obtain all rules whose antecedent is a subset of the given test data
        predRes = np.array([])
        for row in newdata:
            predLab = self.predictLabels(row, k)
            predRes = np.append(predRes, predLab)
        return predRes

    








