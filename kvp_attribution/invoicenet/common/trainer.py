import time

import numpy as np
import tensorflow as tf

from kvp_attribution.invoicenet.common.model import Model


def train(model: Model,
          train_data: tf.data.Dataset,
          val_data: tf.data.Dataset,
          total_steps=50,
          early_stop_steps=0):
    print_interval = 20
    no_improvement_steps = 0
    best = float("inf")

    train_iter = iter(train_data)
    val_iter = iter(val_data)

    start = time.time()
    for step in range(total_steps):
        try:
            train_loss = model.train_step(next(train_iter))
        except StopIteration:
            print("Couldn't find any training data! Have you prepared your training data?")
            print("Terminating...")
            continue

        if not np.isfinite(train_loss):
            raise ValueError("NaN loss")

        if step % print_interval == 0:
            took = time.time() - start

            try:
                val_loss = model.val_step(next(val_iter))
            except StopIteration:
                print("Couldn't find any validation data! Have you prepared your training data?")
                print("Terminating...")
                continue

            print("[%d/%d | %.2f steps/s]: train loss: %.4f val loss: %.4f" % (
                step, total_steps, (step + 1) / took, train_loss, val_loss))
            if not np.isfinite(val_loss):
                raise ValueError("NaN loss")
            if val_loss < best:
                no_improvement_steps = 0
                best = val_loss
                model.save("best")
            elif early_stop_steps > 0:
                no_improvement_steps += print_interval
                if no_improvement_steps >= early_stop_steps:
                    print("Validation loss has not improved for {} steps, terminating!".format(no_improvement_steps))
                    return
