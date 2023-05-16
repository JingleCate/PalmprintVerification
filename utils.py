import os
import numpy as np
import matplotlib.pyplot as plt

def plot_loss(data, iter):
    # 计算每10个取平均的loss
    avg_losses = []
    for i in range(0, len(data), 10):
        avg_loss = sum(data[i:i+10]) / 10
        avg_losses.append(avg_loss)

    # 创建一个新的图像
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制曲线图
    x = range(len(avg_losses))
    ax.plot(x, avg_losses)

    # 添加标题和标签
    ax.set_title('Training Loss')
    ax.set_xlabel('iteration')
    ax.set_ylabel('Loss')

    # 添加网格线
    # ax.grid(True)

    # 显示图像
    plt.savefig("./imgs/loss-it%d.png" % iter)
    # plt.show()
    plt.close()
    
    


def plot_evaluate(accuracy, precision, recall, f1, epoch):
    # 创建一个新的图像
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制曲线图
    x = np.arange(len(accuracy))
    ax.plot(x, accuracy, label='Accuracy')
    ax.plot(x, precision, label='Precision')
    ax.plot(x, recall, label='Recall')
    ax.plot(x, f1, label='F1 Score')

    # 添加图例
    ax.legend()

    # 添加标题和标签
    ax.set_title('Model Evaluation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')

    # 添加网格线
    # ax.grid(True)


    # 显示图像
    plt.savefig("./imgs/epoch%d-%dindex.png" % (epoch))
    # plt.show()
    plt.close()
    

if __name__ == "__main__":
    accuracy = [0.8, 0.9, 0.85, 0.92, 0.88, 0.93, 0.91, 0.95]
    precision = [0.75, 0.85, 0.80, 0.90, 0.87, 0.92, 0.89, 0.94]
    recall = [0.70, 0.80, 0.75, 0.85, 0.82, 0.87, 0.84, 0.89]
    f1 = [0.72, 0.82, 0.77, 0.87, 0.84, 0.89, 0.86, 0.91]


    plot_evaluate(accuracy, precision, recall, f1, 0)
