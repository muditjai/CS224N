import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    N-dimensional vector (treat the vector as a single row) and
    for M x N matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        ### YOUR CODE HERE
        x -= np.max(x, axis=1)[:, np.newaxis]
        x_exp = np.exp(x)
        x = x_exp / np.sum(x_exp, axis=1)[:, np.newaxis]
        ### END YOUR CODE
    else:
        # Vector
        ### YOUR CODE HERE
        x -= np.max(x)
        x_exp = np.exp(x)
        x = x_exp / np.sum(x_exp)
        ### END YOUR CODE

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print "You should be able to verify these results by hand!\n"


def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE

    # Equal softmax result rows.
    test1 = np.array([[1, 2, 3], [4, 5, 6]])
    ans1 = np.array([[0.09003057,  0.24472847,  0.66524096], [0.09003057,  0.24472847,  0.66524096]])
    assert np.allclose(ans1, softmax(test1), rtol=1e-05, atol=1e-6)

    # General matrix test
    test2 = np.array([[1, 2, 3, 4.5],
                      [4, 5, 6, 1000],
                      [-3.1, -np.pi, -0.04, np.e]])
    ans2 = np.array([[0.02261278,  0.0614679 ,  0.16708706,  0.74883227],
                     [0.,  0.,  0.,  1.],
                     [0.00278025,  0.00266698,  0.05929586,  0.93525692]])
    assert np.allclose(ans2, softmax(test2), rtol=1e-05, atol=1e-6)

    # 1 row matrix test
    test3 = np.array([[-3.1, -np.pi, -0.04, np.e]])
    ans3 = np.array([[0.00278025, 0.00266698, 0.05929586, 0.93525692]])
    assert np.allclose(ans3, softmax(test3), rtol=1e-05, atol=1e-6)

    # Vector test
    test4 = np.array([-3.1, -np.pi, -0.04, np.e])
    ans4 = np.array([0.00278025, 0.00266698, 0.05929586, 0.93525692])
    assert np.allclose(ans4, softmax(test4), rtol=1e-05, atol=1e-6)

    ### END YOUR CODE

    print "Tests passed."


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
