import os

if __name__ == "__main__":
    # Check number of trained models
    with open('model_trained.txt', 'w') as f:
        if not os.path.exists(os.path.join(os.getcwd(), 'output')):
            f.write("0")                    # 0 models have been trained
        else:
            f.write(str(len(os.listdir(os.path.join(os.getcwd(), 'output')))))
    f.close()
