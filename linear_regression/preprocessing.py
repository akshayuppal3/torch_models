def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        # train model
        model.train()

        # make prediction
        yhat = model(x)

        # compute the loss
        loss = loss_fn(y, yhat)

        # compute the gradients
        loss.sum().backward()

        # Update the parametes and zero gradients
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    # return function that will be called inside train loop
    return train_step
