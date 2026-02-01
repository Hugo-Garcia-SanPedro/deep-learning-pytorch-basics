import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential (
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # Output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        # The output of the last layer are called logits
        logits = self.layers(x)
        return logits
    
X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
Y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
Y_test = torch.tensor([0, 1])

class ToyDataset(Dataset):
    def __init__(self, X, Y):
        self.features = X
        self.labels = Y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
    
    def __len__(self):
        return self.labels.shape[0]
    
train_ds = ToyDataset(X_train, Y_train)
test_ds = ToyDataset(X_test, Y_test)

torch.manual_seed(123)

train_loader = DataLoader (
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0 # Background processes
)

test_loader = DataLoader (
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.5
)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)

        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                f" | Train Loss: {loss:.2f}")
        
    model.eval()
    with torch.no_grad():
        outputs = model(X_train)
    print(outputs)

def compute_accuracy(model, dataloder):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloder):
        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()

print(compute_accuracy(model, train_loader))

torch.save(model.state_dict(), "model.pth")