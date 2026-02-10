## EX:01 Neural-Network-Regression-Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

<img width="954" height="633" alt="image" src="https://github.com/user-attachments/assets/8d9ac77c-869d-4d23-a4fe-49621970ff2a" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: JAGANNIVASH U M
### Register Number: 212224240059
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}
  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here since it's a regression task
        return x
# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```
## Dataset Information

<img width="600" height="578" alt="image" src="https://github.com/user-attachments/assets/7314a0e3-ef61-4f1e-a2e7-a16f12f34825" />



## OUTPUT

<img width="314" height="448" alt="image" src="https://github.com/user-attachments/assets/82778078-89f8-4dc5-ad76-0ac76a2b60c5" />






<img width="565" height="325" alt="Screenshot 2026-02-10 153104" src="https://github.com/user-attachments/assets/28d6fd1c-8446-4200-83a5-2bfb84fd1c27" />

### Training Loss Vs Iteration Plot

<img width="961" height="671" alt="image" src="https://github.com/user-attachments/assets/9cdeb81d-dd2e-41e8-bdde-4978cdfca0ed" />



### New Sample Data Prediction


<img width="1325" height="235" alt="image" src="https://github.com/user-attachments/assets/6279d3e2-cfaa-4182-8d66-07799610888a" />




## RESULT:

Thus,the code was successfully executed  to develop a neural network regression model...

