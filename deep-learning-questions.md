## Q: What are skip connections and why is it useful? What are some example model architectures?
Skip connections were introduced in ResNets to allow an alternate pathway for info/gradients to propagate in a model. It helps to prevent loss of information in deep networks and prevents vanishing and exploding gradients. Skip connections fit a residual mapping, so learn delta between layers.

## Q: What is a vanishing gradient?
It is the phenomena that occurs in deep neural nets where the gradients reduce to 0, making it difficult to train a model. Certain activation functions attempt to map a large input space to a small output space [0,1] which causes small derivatives and therefore a gradient that vanishes to 0.

## Q: What is max pooling and why is it used?
It is a downsampling operation that allows a layer to represent a larger effective receptive field (spatial resolution). Essentially, it acts as a dimensionality reduction and give a compressed representation of an input. It increases the receptive field and allows for some robustness in that the output can be unchanged even if input changes. No (free) parameters learned. 

## Q: Difference between ReLU and Leaky ReLU. What's the most common issue with ReLU?
Both are activation functions. In ReLU, negative inputs are squashed to 0 and positive inputs get mapped to the same output. Leaky ReLU is an improved version of ReLU that helps with sparse gradients by having a small output for a negative input. The most common issue with ReLU is the dying ReLU problem wherein that it can cause weights to not update and cause gradients to forever be 0 at a node i.e. dead nodes that never activate. An advantage to using ReLU as opposed to other activation functions is that it does not suffer from the vanishing gradient problem and it's easier to calculate the forward and backward pass (i.e. it's gradient is simpler, either 0 or 1). 
- leaky ReLU: f(x)=max(0.01*x , x)

## Q: How does dropout work in training and inference?
During training, dropout will drop a certain percentage of nodes (randomly) from activating and setting it to 0. This prevents nodes from co-adapting too much. In inference, the output is scaled appropiately to account for all nodes being activated.
- training: H1 = np.maximum(0, np.dot(W1, X) + b1)
- inference: H1 = np.maximum(0, np.dot(W1, X) + b1) * p

## Q: L1 vs L2 regularization
Regularization helps prevent a model from overfitting to the training set, which causes the network to fail to generalize to new data. L1 is more robust to outliers and favors sparse inputs.

## Q: What are some ways to prevent overfitting?
- regularization
- data augmentation
- dropout
- add noise to inputs
- batch normalization

## Q: What is batch normalization? How do you apply it in training and testing?
Introduces a covariance shift by normalizing the activation (zero-mean, unit-variance) based on current mini batch. It forces inputs to be nicely scaled at each layer which improves gradient flow. Freeze batch norning during testing to prevent learned parameters from changing i.e. don't change the scale and shift params. It acts as a implicit regularization during training.

## Q: What's the difference between SGD (stochastic gradient descent) and batch gradient descent?
SGD updates the weights/parameters for each training example, whereass batch gradient descent updates the weights after a batch of examples. 

## Q: What is the runtime of multiplication of two matrices?
O(N^3) - the dot product is O(N) and you have to do it for NxN matrices.

## Notes
- CNNs allow for spatial invariance i.e. an object can be shifted and translated in an image and it will still represent the object. This allows for a model to not attach an object to a certain location in an image. Also allows for fewer parameters and weight sharing. It processes multidimensional arrays. 
- precision: number of detections that are actually correct
    precision = TP / (TP + FP)
- recall: number of ground truth objects that are actually detected
    recall = TP / (TP + FN)
- object detection
    - two stage models: selective search in first stage to find candidate objects (region proposals). this helps reduce false positives, usually the feature maps produces contains an object.
- gradient: change in loss wrt to change in weights/bias
