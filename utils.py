import matplotlib.pyplot as plt

# fancy graphics
plt.style.use('ggplot')

def plotLoss(losses):
    #Plotting the validation and training errors
    x_axis = range(len(losses['loss']))
    plt.plot(x_axis, losses['val_loss'], label='Validation loss', linestyle='--')
    plt.plot(x_axis, losses['loss'], label='Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def nextAction():
    prompt = """
Enter one of 1, 2, 3:
\t1) repeat process
\t2) plot training and validation loss over epochs
\t3) save encoder weights
"""
    action = input(prompt)
    while action not in ['1','2','3']:
        action = input("Wrong input, enter again: ")

    return int(action)
