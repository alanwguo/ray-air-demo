import ray
from ray import train
from ray.air import session, Checkpoint
from ray.train.torch import TorchCheckpoint
from ray.util import placement_group
import torchvision
import torchvision.transforms as transforms
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_dataset = torchvision.datasets.CIFAR10("data", download=True, train=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10("data", download=True, train=False, transform=transform)

train_dataset: ray.data.Dataset = ray.data.from_torch(train_dataset)
test_dataset: ray.data.Dataset = ray.data.from_torch(test_dataset)

def convert_batch_to_numpy(batch: List[Tuple[torch.Tensor, int]]) -> Dict[str, np.ndarray]:
    images = np.array([image.numpy() for image, _ in batch])
    labels = np.array([label for _, label in batch])
    return {"image": images, "label": labels}


train_dataset = train_dataset.map_batches(convert_batch_to_numpy)
test_dataset = test_dataset.map_batches(convert_batch_to_numpy)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_loop_per_worker(config):
    model = train.torch.prepare_model(Net())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_dataset_shard = session.get_dataset_shard("train")

    ray.get(do_train_step.options(placement_group=None).remote())

    for epoch in range(5):
        running_loss = 0.0
        train_dataset_batches = train_dataset_shard.iter_torch_batches(
            batch_size=config["batch_size"],
        )
        for i, batch in enumerate(train_dataset_batches):
            # get the inputs and labels
            inputs, labels = batch["image"], batch["label"]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

        metrics = dict(running_loss=running_loss)
        checkpoint = TorchCheckpoint.from_state_dict(model.module.state_dict())
        session.report(metrics, checkpoint=checkpoint)


@ray.remote(num_gpus=1)
def do_train_step():
    if not torch.cuda.is_available():
        print("No GPU detected.")
        raise RuntimeError("No GPU detected.")


from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig

trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={"batch_size": 2},
    datasets={"train": train_dataset},
    scaling_config=ScalingConfig(num_workers=4),
)

from ray import tune
from ray.tune import Tuner
tuner = Tuner(
    trainer,
    # Add some parameters to tune
    param_space={"train_loop_config": {"batch_size": tune.choice([1, 2, 3, 4, 5])}},
    # Specify tuning behavior
    tune_config=tune.TuneConfig(metric="running_loss", mode="min", num_samples=2),
)
result = tuner.fit()

