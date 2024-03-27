import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the loan data from CSV file
file_path = "csv/loan_data.csv"
loan_data = pd.read_csv(file_path)
# Check if a file was selected
if file_path:
    loan_data = pd.read_csv(file_path)
    print("Loan data loaded successfully.")
else:
    print("No file selected. Exiting.")

# Drops the Loan_ID column as it is not used in loan predictions
loan_data = loan_data.drop(columns=['Loan_ID'])

# Convert string entries to numerical using label encoding
label_encoders = {}
for column in loan_data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    loan_data[column] = label_encoders[column].fit_transform(loan_data[column])

# Handles the missing values and replaces them with the median for numerical columns
loan_data.fillna(loan_data.median(), inplace=True)

# Split data into features and target variable
X = loan_data.drop(columns=['Loan_Status'])
y = loan_data['Loan_Status']

# Train Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)


# Prompts user for new loan data
def get_new_loan_data(label_encoders):
    column_prompts = {
        'Gender': "Enter the applicant's gender. Acceptable values: 'Male' or 'Female'",
        'Married': "Enter the applicant's marital status. Acceptable values: Enter 'Yes' or 'No'",
        'Dependents': "Enter the applicant's number of dependents. Acceptable values: '0', '1', '2', '3+'",
        'Education': "Enter the applicant's education: Acceptable values: 'Graduate' or 'Not Graduate'",
        'Self_Employed': "Enter if the applicant is self employed: Acceptable values: 'Yes' or 'No'",
        'ApplicantIncome': "Enter the applicant's gross monthly income: Acceptable values: integer",
        'CoapplicantIncome': "Enter the co-applicant's income. Acceptable values: integer",
        'LoanAmount': "Enter the loan amount (in thousands): Acceptable values: integer",
        'Loan_Amount_Term': "Enter the loan term (in months): Acceptable values: integer",
        'Credit_History': "Enter if the applicant has credit history: Acceptable values: '1' for Yes or '0' for No",
        'Property_Area': "Enter the applicant's property area: Acceptable values: 'Rural', 'Semiurban', or 'Urban'"
    }

    new_data = {}
    for column in X.columns:
        if column != 'Loan_Status':
            if column in label_encoders:
                while True:
                    print(column_prompts[column])
                    value = input(f"New applicant data for {column}: ")
                    try:
                        encoded_value = label_encoders[column].transform([value])[0]
                        new_data[column] = encoded_value
                        break
                    except ValueError:
                        print("Error: Please enter a valid value.")
            else:
                while True:
                    print(column_prompts[column])
                    value = input(f"New applicant data for {column}: ")
                    try:
                        new_data[column] = float(value)  # Convert to float for numerical columns
                        break
                    except ValueError:
                        print("Error: Please enter a valid numerical value.")
    return new_data


# Allows user to select visualization
def explore_dataset(loan_data):
    print("Choose a visualization:")
    print("1. Histogram of Loan Amount")
    print("2. Pie chart of Graduates and Non-graduates among Accepted Loans")
    print("3. Bar chart of Loan Status")
    print("4. Correlation Heatmap")
    print("5. Quit")
    choice = input("Enter your choice (1/2/3/4/5): ")

    if choice == '1':
        loan_data['LoanAmount'].hist(bins=20)
        plt.title('Distribution of Loan Amount')
        plt.xlabel('Loan Amount (Thousands)')
        plt.ylabel('Frequency')
        plt.show()
    elif choice == '2':
        # Selecting only the accepted loans
        accepted_loans = loan_data[loan_data['Loan_Status'] == 1]

        # Counting the number of graduates and non-graduates among accepted loans
        graduate_counts = accepted_loans[accepted_loans['Education'] == 0].shape[0]
        non_graduate_counts = accepted_loans[accepted_loans['Education'] == 1].shape[0]

        # Creating a pie chart
        labels = ['Graduates', 'Non-graduates']
        sizes = [graduate_counts, non_graduate_counts]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.title('Distribution of Graduates and Non-graduates among Accepted Loans')
        plt.show()

    elif choice == '3':
        # Map numerical labels to status labels
        loan_data['Loan_Status_Label'] = loan_data['Loan_Status'].map({1: 'Approved', 0: 'Declined'})

        # Plot bar chart with customized colors
        colors = ['green', 'red']
        loan_data['Loan_Status_Label'].value_counts().plot(kind='bar', rot=0, color=colors)

        # Customize plot labels
        plt.title('Bar chart of Loan Status')
        plt.xlabel('Loan Status')
        plt.ylabel('Count')
        plt.xticks(range(2), ['Approved', 'Declined'])
        plt.show()

    elif choice == '4':
        # Calculate correlation matrix
        corr_matrix = loan_data.corr()

        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
        plt.title('Correlation Heatmap')
        plt.xlabel('Features')
        plt.ylabel('Features')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    elif choice == '5':
        print("Exiting the program...")
        exit()
    else:
        print("Invalid choice.")

print("Loan Prediction Program")

# Prompts user for action
while True:
    action = input("Enter '1' view visualizations of the data, '2' to enter new loan information, or '3' to quit: ")
    if action == '1':
        explore_dataset(loan_data)
    elif action == '2':
        # Prompts user for new loan data
        new_loan_data = get_new_loan_data(label_encoders)

        # Makes a prediction on new loan data
        prediction = clf.predict(pd.DataFrame([new_loan_data]))
        print("Predicted Loan Status:", "Approved" if prediction[0] == 1 else "Declined")

        # Evaluates the accuracy of the model for the new loan
        X_new = pd.DataFrame([new_loan_data])
        y_new = pd.Series([prediction[0]], name='Loan_Status')
        accuracy = clf.score(X_new, y_new) * 100
        print("Model Accuracy for this loan:", f"{accuracy:.2f}%")
    elif action == '3':
        print("Exiting the program...")
        break
    else:
        print("Invalid input. Please enter either '1', '2', or '3'.")