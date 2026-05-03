# HW1 LLM and Discussion Disclosure

## References

- Course lecture slides: `Lecture01_Introduction.pdf`, `Lecture02_NN.pdf`, `Lecture03_BackPropPart1.pdf`, `Lecture04_BackPropPart2.pdf`.
- The HW1 assignment handout, starter code, provided datasets, and provided test files.
- PyTorch documentation:
  - [`torch.autograd`](https://docs.pytorch.org/docs/2.9/autograd.html), especially the role of tensors with `requires_grad`, `backward`, `grad`, and custom `torch.autograd.Function` classes.
  - [`torch.autograd.gradcheck`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html) and the PyTorch gradcheck notes for comparing analytical gradients to finite-difference numerical gradients.
  - [`torch.utils.data.TensorDataset`](https://docs.pytorch.org/docs/stable/data.html) for wrapping feature and label tensors into a dataset.
  - [`torch.nn.Sequential`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html) for constructing a feed-forward network as an ordered sequence of layers.
  - [`torch.optim.SGD`](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD) for understanding the optimizer arguments used by the provided training code, including learning rate, momentum, and weight decay.

## Use of Course Materials

I used the lecture slides as the main conceptual reference for the assignment. The early neural-network lectures supported my understanding of fully connected layers, nonlinear activation functions, classification outputs, loss functions, and training with gradient-based optimization. The backpropagation lectures were especially useful for reasoning about the chain rule, upstream gradients, and why the backward methods must return gradients with the same shapes as their corresponding inputs.

## LLM Disclosure

I used LLMs as support tools while working through this assignment. My prompts focused on understanding the assignment instructions, the tensor shapes in the forward and backward passes, checking how finite-difference gradient tests relate to analytical gradients, and interpreting PyTorch warnings and test outputs. The LLM responses helped explain concepts such as `ctx.save_for_backward`, `torch.autograd.grad`, `gradcheck`, TensorDataset preprocessing, and how to verify saved `.pt` result files and trained models.

For the learning tasks, I used the assignment constraints and lecture concepts to guide the model setup, then used the LLM to sanity-check whether the chosen settings satisfied the written requirements and to interpret training accuracy/loss results. I did not treat LLM responses as a substitute for understanding the homework: I implemented functions, checked the formulas against the handout, ran the provided tests, trained the models, tuned the hyperparameters, verified the saved result files, and evaluated the XOR and Iris model performances myself. Any debugging suggestions from the LLM were reviewed in the context of the assignment instructions and validated by running the required tests.
