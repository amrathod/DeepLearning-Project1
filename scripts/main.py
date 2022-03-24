from unittest.mock import patch
from project1_model import *
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def train(net, criterion, optimizer, trainloader, epoch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    train_acc = (correct / total)*100.0
    train_loss = train_loss/len(trainloader)
    print(f"+++++++++++ Starting Epoch: {epoch} +++++++++++")
    print("")
    print(f"Training Loss: {train_loss}, Training Accuracy: {train_acc}")
    return train_loss, train_acc

def test(net, criterion, testloader, epoch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        test_acc = correct / total * 100.0
        test_loss = test_loss / len(testloader)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
    return test_loss, test_acc

def main(max_epochs, use_saved_model, check_accuracy_by_class):
    """
    max_epochs: maximum epochs to run if early stopping is not used
    used_saved_model: boolean to load saved model to evaluate or continue training
    check_accuracy_by_class: boolean that triggers printing of accuracy results by class 
    """

    transform_train = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomPerspective(0.3, 0.5),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

                                        ])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='../CIFAR10/', train=True, 
                                        download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, 
                                        shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='../CIFAR10/', train=False, 
                                            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=150, 
                                            shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = project1_model()
    net = net.to(device)

    train_loss_list = [] # holds total loss of each epoch
    test_loss_list = []
    train_accuracy_list = [] # accuracy of each epoch
    test_accuracy_list = []
    best_acc = 0.0

    # The patience left variable is decremented by one every time the test loss of an epoch is higher 
    # than the best test loss. Once it reaches zero, training is stopped to avoid overfitting
    patience_left = 15 
    epoch = 1
    best_loss = 5.0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr= 0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if use_saved_model:
        print('Using Saved Model..')
        saved_model = torch.load('./saved_model/resnet_adam_dropout_4_9M.pth')
        net.load_state_dict(saved_model['net'])
        best_acc = saved_model['accuracy']
        best_loss = saved_model['loss']

    while patience_left > 0 and epoch < max_epochs:    

        train_loss, train_acc = train(net=net, criterion=criterion, 
                                    optimizer=optimizer, trainloader=trainloader, epoch=epoch)
        test_loss, test_acc = test(net=net, criterion=criterion, testloader=testloader, epoch=epoch)
        
        train_accuracy_list.append(train_acc)
        train_loss_list.append(train_loss)
        test_accuracy_list.append(test_acc)
        test_loss_list.append(test_loss)

        print(f"Patience Remaining: {patience_left}")

        if best_loss < test_loss:
            patience_left -= 1
            print("")
            print("\x1b[6;30;43m" + "Test Loss Increased" "\x1b[0m")
        else: 
            best_loss = test_loss
        
        if best_acc < test_acc:
            best_acc = test_acc

            state = {
            'net': net.state_dict(),
            'accuracy': best_acc,
            'loss': best_loss,
        }
        if not os.path.isdir('saved_model'):
            os.mkdir('saved_model')
        torch.save(state, './saved_model/resnet_adam_dropout_4_9M.pth')

        print(f"Best Test Loss: {best_loss}")
        print(f"Best Test Accuracy: {best_acc}")
        print("")
        scheduler.step()
        epoch += 1

    print("\x1b[6;30;41m" + "Early Stopping to Avoid Overfitting" "\x1b[0m")

    if check_accuracy_by_class:
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images.to(device))
                _, predictions = outputs.max(1)
                for label, prediction in zip(labels.to(device), predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1
        print("")
        print("+++++++++ " + "\x1b[6;30;42m" + "Accuracy By Class" "\x1b[0m" + " ++++++++")
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        print("")   

    return train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list

if __name__ == "__main__":
    load_model = False # Change to true to use saved model 
    epochs_to_run = 50 # to test a saved model change to 1
    generate_plots = False
    check_accuracy_by_class = True
    train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list = main(max_epochs=epochs_to_run, 
                                                                                use_saved_model=load_model,
                                                                                check_accuracy_by_class=check_accuracy_by_class)

    if generate_plots:
        loss_figure_name = "loss.png"
        accuracy_figure_name = "accuracy.png"
        plt.rcParams["figure.figsize"] = (19.20, 10.80)
        font = {"family" : "sans",
                "weight" : "normal",
                "size"   : 26}
        matplotlib.rc("font", **font)
        matplotlib.rcParams['lines.linewidth'] = 3
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42

        plt.figure()
        plt.title("Loss - 4.9M Parameters")
        plt.plot(train_loss_list,  color="r", label="Training Loss")
        plt.plot(test_loss_list,  color="b", label="Test Loss")
        plt.grid()
        plt.legend()
        plt.savefig("./figures/" + loss_figure_name)
        plt.close()

        plt.figure()
        plt.title("Accuracy - 4.9M Parameters")
        plt.plot(train_accuracy_list,  color="r", label="Train")
        plt.plot(test_accuracy_list,  color="b", label="Test")
        plt.grid()
        plt.legend()
        plt.savefig("./figures/" + accuracy_figure_name)
        plt.close()

    
        