import joblib

# Load the trained model
model = joblib.load("legal_model.pkl")

print("📚 Legal Citation Classifier")
print("-----------------------------")
while True:
    user_input = input("\n📝 Enter a legal citation sentence (or type 'exit' to quit):\n> ")
    if user_input.lower() == 'exit':
        break

    # Predict the class
    prediction = model.predict([user_input])[0]
    print(f"📌 Predicted Class: {prediction}")
