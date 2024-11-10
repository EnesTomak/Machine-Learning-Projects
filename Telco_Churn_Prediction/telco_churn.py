import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
from helpers import *

warnings.simplefilter(action="ignore")

# Set display options for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load dataset
df = pd.read_csv("datasets/Telco-Customer-Churn.csv")

# Convert 'TotalCharges' to numeric and handle 'Churn' encoding
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

def preprocess_telco_data(df):
    # Data checks and feature engineering
    check_df(df)

    # Identify categorical and numerical columns
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    # Summarize categorical features
    for col in cat_cols:
        cat_summary(df, col)

    # Summarize numerical features
    for col in num_cols:
        num_summary(df, col)

    # Missing values summary
    missing_columns = missing_values_table(df, na_name=True)

    # Feature engineering
    df["NEW_TENURE_YEAR"] = pd.cut(df["tenure"],
                                    bins=[-1, 12, 24, 36, 48, 60, 100],
                                    labels=["0-1 Year", "1-2 Year", "2-3 Year", "3-4 Year", "4-5 Year", "5-6 Year"])
    df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)
    df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)
    df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)
    df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)
    df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)
    df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0)
    df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

    # Update categorical and numerical columns
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    # Label encoding for binary categorical features
    binary_cols = [col for col in cat_cols if df[col].nunique() == 2]
    for col in binary_cols:
        df = label_encoder(df, col)

    # One-hot encoding for other categorical features
    cat_cols = [col for col in cat_cols if col not in binary_cols]
    df = one_hot_encoder(df, cat_cols, drop_first=True)

    # Handle outliers
    for col in num_cols:
        if check_outlier(df, col):
            replace_with_thresholds(df, col)

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])
    return df

# Define the neural network class
class TelcoChurnNN(nn.Module):
    def __init__(self, input_size):
        super(TelcoChurnNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        if torch.any(torch.isnan(x)):
            print("NaN found after layer 1")
        x = self.relu(self.layer2(x))
        if torch.any(torch.isnan(x)):
            print("NaN found after layer 2")
        x = self.sigmoid(self.layer3(x))
        if torch.any(torch.isnan(x)):
            print("NaN found after output layer")
        return x



# Preprocess the data
df = preprocess_telco_data(df)
X = df.drop("Churn", axis=1).values
y = df["Churn"].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Reshaped for output layer
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create DataLoader for training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the neural network, loss function, and optimizer
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

input_size = X_train.shape[1]
model = TelcoChurnNN(input_size)
model.apply(initialize_weights)  # Apply the weight initialization
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adjust learning rate

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)

        # Debugging print
        print("Model Outputs:", outputs.detach().numpy().flatten())  # Check output values

        # Ensure outputs are within the correct range
        if not torch.all((outputs >= 0) & (outputs <= 1)):
            print("Error: Outputs out of bounds!")  # Alert if any output is out of range
            print("Invalid Outputs:", outputs)  # Print the invalid outputs for debugging
            continue  # Skip this batch

        # Calculate the loss
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        running_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Evaluation on test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    predicted = (test_outputs > 0.5).float()
    accuracy = (predicted.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item()
    print(f"Test Accuracy: {accuracy:.4f}")
