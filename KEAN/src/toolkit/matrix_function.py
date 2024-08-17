import numpy as np

def calculate_test_distribition(test_confusion):
    positive = test_confusion[0,1]+test_confusion[1,1]
    negative = test_confusion[0,0]+test_confusion[1,0]
    ans = np.array([negative,positive]).T
    sum = np.sum(ans)
    ans = np.divide(ans,sum)
    print(ans.shape)
    return ans 

if __name__ == "__main__":
    train_confusion = np.shape(2,2) # the train_confusion matrix
    test_confusion = np.shape(2,2) # the test_confusion matrix
    sum = np.sum(train_confusion)
    train_confusion = np.divide(train_confusion.T,sum)
    print("train sum:",sum)
    print("train conf:",train_confusion)
    ans = calculate_test_distribition(test_confusion)
    train_inv = np.linalg.inv(train_confusion)
    print(train_inv)
    print(ans)
    # final denotes the $\mathcal{w}$ for re-training
    final = np.dot(train_inv,ans)
    print(final)
    