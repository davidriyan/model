import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data_path = "C:/Users/HP/Documents/project_ml/Survei_Pengalaman_Pembelian_Hijab_Online.xlsx"
df = pd.read_excel(data_path)

# Print the column names to verify the structure of the DataFrame
print("Column Names:", df.columns)

# Preprocess data (You may need to handle missing values, encode categorical variables, etc.)

# Extract features and target
X = df[['Usia', 'Status', 'Apa yang menjadi bahan pertimbanganmu ketika membeli hijab?', 'Apakah kamu lebih menyukai pembelian hijab secara online atau offline?',
        'Apa yang menjadi alasan utamamu lebih menyukai metode tersebut?', 'Apa kesulitan yang kamu hadapi ketika membeli jilbab secara online?', 'Jika terdapat aplikasi yang dapat memungkinkan kamu mencoba jilbab suatu brand, apakah menurutmu itu akan membantu?', 'Mengapa kamu berpikir demikian', 'Apa saja brand hijab yang kamu ketahui?', 'Apakah kamu mengetahui brand hijab Bohemian Era (bohemian.eraa)?']]
# Adjust the target column accordingly
y = df['Apa yang menjadi bahan pertimbanganmu ketika membeli hijab?'] = df[
    'Apa yang menjadi bahan pertimbanganmu ketika membeli hijab?'].astype(str)


# Encode categorical variables if needed
X_encoded = pd.get_dummies(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42)

# Build and train a RandomForestClassifier (you can choose a different model based on your needs)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Save the model (you can use other serialization methods based on your integration requirements)
joblib.dump(model, 'your_model.pkl')
