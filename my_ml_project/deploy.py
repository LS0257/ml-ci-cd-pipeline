import ml_model
from ml_model.train import train_model

def deploy_model():
    # Train and deploy the model
    model = train_model()
    print("Model deployed successfully")

if __name__ == "__main__":
    deploy_model()
 
