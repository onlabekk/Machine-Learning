import pandas as pd
import numpy as np
from collections import defaultdict

def KL(p, q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    if p == 0:
        return - np.log(1 - q)
    if p == 1:
        return - np.log(q)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

class Node():
    def __init__(self, y_mean, ntotal):
        self.y_mean = y_mean
        self.ntotal = ntotal
        
class DecisionTree():
    def __init__(self, split=None, rightBranch=None, leftBranch=None, node=None, result=None):
        self.split = split
        self.rightBranch = rightBranch
        self.leftBranch = leftBranch
        self.node = node
        self.result = result
        
class UpliftTree():
    def __init__(self, max_depth, scoring='KL', min_samples_leaf=100, min_samples_treatment=10, n_rand_features=None, random_state=None, rng=None, norm=True):
        self.max_depth = max_depth
        self.n_rand_features = n_rand_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        self.feature_imp_dict = {}
        self.norm = norm
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState(random_state)
        self.scoring = scoring
        if scoring == 'KL':
            self.score_fu = self.evaluate_KL
        if scoring == 'ED':
            self.score_fu = self.evaluate_ED
        if scoring == 'Chi':
            self.score_fu = self.evaluate_Xi
        self.fitted_uplift_tree = None
    
    def split_data(self, X, y, w, feature_idx, val):
        if isinstance(val, float) or isinstance(val, int):
            idx = X[:, feature_idx] >= val
        elif isinstance(val, str):
            idx = X[:, feature_idx] == val
        else:
            print("Val is not appropriate")
            return False
        return X[idx], X[~idx], w[idx], w[~idx], y[idx], y[~idx]
    
    def choose_features(self, n_rand): ##########
#         n_randf = 7
        if n_rand < self.n_features:
            f = self.rng.choice(range(self.n_features), replace=False, size=n_rand)
            return f
        else:
            return range(self.n_features)
        
    def gini(self, p):
        return 1 - p*p - (1 - p)*(1 - p)
    
    def entropy(self, p):
        if p == 1 or p == 0:
            return 0
        return -p*np.log(p) - (1-p)*np.log(1-p)
        
    def evaluate_KL(self, node):
        if node.ntotal[0] == 0:
            return 0
        return KL(node.y_mean[1], node.y_mean[0])
        
    def evaluate_ED(self, node):
        if node.ntotal[0] == 0:
            return 0
        return 2 * (node.y_mean[1] - node.y_mean[0])**2
        
    def evaluate_Xi(self, node):
        if node.ntotal[0] == 0:
            return 0
        q = np.clip(node.y_mean[0], 1e-5, 1 - 1e-5)
        return (node.y_mean[1] - q)**2 / q + (node.y_mean[1] - q)**2 / (1 - q)
    
    def normalize(self, node, left_node):
        nc = node.ntotal[0]
        nt = node.ntotal[1]
        nc_left = left_node.ntotal[0]
        nt_left = left_node.ntotal[1]
        y_c = np.clip(nc_left/nc, 1e-5, 1 - 1e-5)
        y_t = np.clip(nt_left/nt, 1e-5, 1 - 1e-5)
        if self.scoring == 'KL':
            norm = self.entropy(nc / (nc + nt)) * KL(y_c, y_t) + (nc / (nc + nt)) * self.entropy(y_c) + (nt / (nc + nt)) * self.entropy(y_t) + 1/2
        if self.scoring == 'ED':
            norm = self.entropy(nc / (nc + nt)) * 2 * (y_c - y_t)**2 + (nc / (nc + nt)) * self.entropy(y_c) + (nt / (nc + nt)) * self.entropy(y_t) + 1/2
        if self.scoring == 'Chi':
            norm = self.entropy(nc / (nc + nt)) * ((y_t - y_c)**2 / y_c + (y_t - y_c)**2 / (1 - y_c)) + (nc / (nc + nt)) * self.entropy(y_c) + (nt / (nc + nt)) * self.entropy(y_c) + 1/2

        return norm
        
    def node_info(self, y, w, min_n_treat=10, regular=100, ParentNode=None):
        
        y_mean, ntotal = {}, {}
        for group in [0, 1]:
            treat_idx = w == group
            ntotal[group] = treat_idx.sum()
            y_sum = y[treat_idx].sum()
            if ParentNode is None:
                y_mean[group] = y_sum / ntotal[group]

            elif len(y) > min_n_treat:
                y_mean[group] = (y_sum + regular * ParentNode.y_mean[group]) / (ntotal[group] + regular)
            
            else:
                y_mean[group] = ParentNode.y_mean[group]
        return Node(y_mean=y_mean, ntotal=ntotal)
        
    def node_fitting(self, X, y, w, depth, n_randf, ParentNode=None):
        
        if len(X) == 0:
            return DecisionTree()
        
        current_node = self.node_info(y, w)
        current_node_score = self.score_fu(current_node)
        
        best_gain = 0
        best_split = None
        
        for feature_idx in self.choose_features(n_randf):
            column = X[:, feature_idx]
            val = column[0]
            if isinstance(val, float) or isinstance(val, int):
                uniq_val = np.unique(column)
                if len(uniq_val) > 10:
                    split_vals = np.percentile(column, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
                else:
                    split_vals = np.percentile(column, [10, 50, 90])
                split_vals = np.unique(split_vals)
            else:
                split_vals = np.unique(column)

            gains = []

            for val in split_vals:
                X_l, X_r, w_l, w_r, y_l, y_r = self.split_data(X, y, w, feature_idx, val)
#                 print(X_r.shape, y_r.shape, w_r.shape, X_l.shape, y_l.shape, w_l.shape)
                if len(y_r) < self.min_samples_leaf or len(y_l) < self.min_samples_leaf:
                    continue
                
                leftNode = self.node_info(y_l, w_l, ParentNode=current_node)
                rightNode = self.node_info(y_r, w_r, ParentNode=current_node)             
#                 print(leftNode.y_mean, leftNode.ntotal, rightNode.y_mean, rightNode.ntotal, self.evaluate_KL(leftNode))
#                 if leftNode.ntotal[0] < self.min_samples_treatment or leftNode.ntotal[1] < self.min_samples_treatment:
#                     continue
#                 if rightNode.ntotal[0] < self.min_samples_treatment or rightNode.ntotal[1] < self.min_samples_treatment:
#                     continue
                            
                left_score = self.score_fu(leftNode)
                right_score = self.score_fu(rightNode)
                l_fraction = len(y_l) / len(y)
                gain =  left_score * l_fraction + right_score * (1 - l_fraction) - current_node_score
                gain_for_imp = left_score * len(X_l) + right_score * len(X_r) - current_node_score * len(X)
                if self.norm == True:
                    gain /= self.normalize(current_node, leftNode)
                 
                #print(gain, left_score, right_score, current_node_score )
                if (gain > best_gain and len(y_r) > self.min_samples_leaf and len(y_l) > self.min_samples_leaf):
#                     print(gain, left_score, right_score, current_node_score, val, feature_idx)
#                     if len(y_r) < self.min_samples_leaf:
#                         print(gain, len(y_r))
                    best_gain = gain
                    best_split = (feature_idx, val)
                    set_left = [X_l, y_l, w_l]
                    set_right = [X_r, y_r, w_r]
                    self.feature_imp_dict[feature_idx] += gain_for_imp
                    
#         print("Best", best_gain, best_split, len(y_r), len(y_l), self.min_samples_leaf) 
        if best_gain > 0 and depth < self.max_depth:
            left_branch = self.node_fitting(*set_left, n_randf=n_randf, depth=depth+1, ParentNode=current_node)
            right_branch = self.node_fitting(*set_right, n_randf=n_randf, depth=depth+1, ParentNode=current_node)
            return DecisionTree(
                split=best_split,
                rightBranch=right_branch, leftBranch=left_branch,
                node=current_node,
            )
        else:
            p = {}
            for group in [0, 1]:
                treat_idx = w == group
                ntotal = np.sum(treat_idx)
                p[group] = round(sum(y[treat_idx]) / ntotal, 6)
#             print(p)
            return DecisionTree(
                node=current_node,
                result=p
            )
        
    def fit(self, X, y, w):
        if isinstance(X, pd.core.frame.DataFrame):
            X, y, w = X.to_numpy(), y.to_numpy(), w.to_numpy()
        self.N, self.n_features = X.shape
        if self.n_rand_features == None:
            self.n_rand_features = max(1, self.n_features // 4)
            
        self.feature_imp_dict = defaultdict(float)
        ParentNode = None
        
        self.fitted_uplift_tree = self.node_fitting(X, y, w, depth=1, n_randf=self.n_rand_features, ParentNode=ParentNode)
        
        self.feature_importances_ = np.zeros(X.shape[1])
        sum_imp = sum(self.feature_imp_dict.values())
        for feature, impurity in self.feature_imp_dict.items():
            self.feature_importances_[feature] = impurity / sum_imp
        
        
    def predict(self, X, full=False):
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.to_numpy()
        ans = np.zeros(len(X))
        prob = np.zeros(len(X))
        prob_control = np.zeros(len(X))
        prob_treatment = np.zeros(len(X))
        for i, elem in enumerate(X):
            mean = self.classify(elem, self.fitted_uplift_tree)
#             print(mean)
            prob_control[i] = mean[0]
            prob_treatment[i] = mean[1]
            class_ = max(mean, key=mean.get)
            prob_ = mean[class_]
            ans[i] = class_
            prob[i] = prob_
        if full:
            return prob_control, prob_treatment
        return ans, prob
        
    def classify(self, x, tree):
        if tree.result is not None:
            return tree.result
        else:
            val = tree.split[1]
            if isinstance(val, str):
                if val == x[int(tree.split[0])]:
                    branch = tree.leftBranch
                else:
                    branch = tree.rightBranch
            elif isinstance(val, float) or isinstance(val, int):
                if val >= x[int(tree.split[0])]:
                    branch = tree.leftBranch
                else:
                    branch = tree.rightBranch
        return self.classify(x, branch)
    
    
class RandomForestUplift():
    """ Uplift Random Forest for Classification.
    Parameters
    ----------
    n_estimators : integer, optional (default=50)
        The number of uplift trees in the uplift random forest.
    max_depth: int, optional (default=10)
        The maximum depth of the uplift tree.
    bootstrapping : bool, optional (default=True)
        The each uplift tree can be either fitted on bootstrapped sample or on whole dataset.
    n_rand_features : int, optional (default=None)
        number of random features for node in each uplift tree when searching the best split.
    scoring : string, optional (default='KL')
        Choose from one of the evaluation metrics: 'KL', 'ED', 'Chi'.
    min_samples_leaf: int, optional (default=100)
        The minimum number of samples required to make split at a leaf node.
    min_samples_treatment: int, optional (default=10)
        The minimum number of samples of treatment group required to make split at a leaf node.
    Outputs
    ----------
    class_: array
        Best treatment group.
    prob: array
        Predicted delta.
    """
    def __init__(self, n_estimators=50, max_depth=10, bootstrapping=True, n_rand_features=None, scoring='KL', min_samples_leaf=100, min_samples_treatment=10, random_state=None, norm=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bootstrapping = bootstrapping
        self.n_rand_features = n_rand_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        self.uplift_forest = []
        self.rng = np.random.RandomState(random_state)
        
        for i in range(n_estimators):
            self.uplift_forest.append(UpliftTree(max_depth = max_depth,
                                                 n_rand_features = n_rand_features,
                                                 min_samples_leaf = min_samples_leaf,
                                                 min_samples_treatment = min_samples_treatment,
                                                 scoring = scoring, rng=self.rng, norm=norm))
        
    
    def bootstrap(self, X, y, w):
        idx = self.rng.choice(range(len(X)), replace=True, size=len(X))
        return X[idx], y[idx], w[idx]
    
    def fit(self, X, y, w):
        if isinstance(X, pd.core.frame.DataFrame):
            X, y, w = X.to_numpy(), y.to_numpy(), w.to_numpy()
        feature_importances = []
        for i in range(self.n_estimators):
            if self.bootstrapping == True:
                X_b, y_b, w_b = self.bootstrap(X, y, w)
                self.uplift_forest[i].fit(X_b, y_b, w_b)
            else:
                self.uplift_forest[i].fit(X, y, w)
            feature_importances.append(self.uplift_forest[i].feature_importances_)
        self.feature_importance_ = np.mean(feature_importances, axis=0)
        self.feature_importance_ /= np.sum(self.feature_importance_)
        
    def predict(self, X):
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.to_numpy()
        trees_control = []
        trees_treatment = []
        for i in range(self.n_estimators):
            c, tr = self.uplift_forest[i].predict(X, full=True)
            trees_control.append(c)
            trees_treatment.append(tr)
        mean_control = np.mean(trees_control, axis=0)
        mean_treatment = np.mean(trees_treatment, axis=0)
        class_ = np.argmax([mean_control,mean_treatment], axis=0)
        prob = np.max([mean_control,mean_treatment], axis=0)
        return class_, prob
    
    
class Boosting():
    """
    Generic class for construction of boosting models
    
    :param n_estimators: int, number of estimators (number of boosting rounds)
    :param base_classifier: callable, a function that creates a weak estimator. Weak estimator should support sample_weight argument
    :param get_alpha: callable, a function, that calculates new alpha given current distribution, prediction of the t-th base estimator,
                      boosting prediction at step (t-1) and actual labels
    :param get_distribution: callable, a function, that calculates samples weights given current distribution, prediction, alphas and actual labels
    """
    def __init__(self, n_estimators=50, base_classifier=None, random_state=None):
        self.n_estimators = n_estimators
        self.base_classifier = base_classifier
        self.rng = np.random.RandomState(random_state)


    def get_alpha(self, y, y_pred_t, distribution):
        """
        Function, which calculates the weights of the linear combination of the classifiers.

        y_pred_t is a prediction of the t-th base classifier
        """

        self.e[self.treatment] = y[self.treatment] != y_pred_t[self.treatment]
        self.e[self.control] = y[self.control] == y_pred_t[self.control]
        self.e.astype(int)
        e_t = sum(distribution[self.treatment][self.e[self.treatment] == 1])/sum(distribution[self.treatment])
        e_c = sum(distribution[self.control][self.e[self.control] == 1])/sum(distribution[self.control])
        
        p_t = sum(distribution[self.treatment])/sum(distribution)
        p_c = sum(distribution[self.control])/sum(distribution)

        alpha = (p_t*e_t + p_c*e_c)/(1 - p_t*e_t + p_c*e_c)


        return alpha, e_t, e_c

    def update_distribution(self, y, y_pred_t, distribution, alpha_t):
        """
        Function, which calculates sample weights

        y_pred_t is a prediction of the t-th base classifier
        """
        ### BEGIN Solution (do not delete this comment)
        indicator = np.ones(len(y))
        y_tr, y_pred_t_tr = y[self.treatment], y_pred_t[self.treatment]
        indicator[self.treatment][y_tr == y_pred_t_tr] = alpha_t
        y_c, y_pred_t_c = y[self.control], y_pred_t[self.control]
        indicator[self.control][1 - y_c == y_pred_t_c] = alpha_t
        
        distribution = distribution*indicator

        ### END Solution (do not delete this comment)

        return distribution

    def fit(self, X, y, w):
        y = np.array(y)
        w = np.array(w)
        n_samples = len(X)
        distribution = np.ones(n_samples, dtype=float)
        self.e = np.zeros(n_samples)
        self.treatment = w == 1
        self.control = w == 0
        self.classifiers = []
        self.alphas = []
        for i in range(self.n_estimators):
            # create a new classifier

            self.classifiers.append(self.base_classifier)
            
            distribution = distribution/sum(distribution)
            self.classifiers[-1].fit(np.multiply(distribution.reshape(1, -1).T, X), y, w)

            y_pred_t, _ = self.classifiers[-1].predict(X)
            y_pred_t = np.array(y_pred_t)

            alpha_t, e_t, e_c = self.get_alpha(y, y_pred_t, distribution)
            self.alphas.append(alpha_t)
            if alpha_t == 1 or e_t >= 0.5 or e_t <= 0 or e_c >= 0.5 or e_c <= 0:
                distribution = self.rng.random(size=n_samples)
                continue
            distribution = self.update_distribution(y, y_pred_t, distribution, alpha_t)


    
    def predict(self, X):
        final_predictions = np.zeros(X.shape[0])
        out = 0
        for s in np.arange(len(self.classifiers)):
            y_pred, _ = self.classifiers[s].predict(X)
            out = out + np.log(1/self.alphas[s])*y_pred
        out -= 1/2*sum(np.log(1/np.array(self.alphas)))
        out = np.sign(out)

        return out