import numpy as np
import matplotlib.pyplot as plt
import torch

c_medium_contrast = {
    "light-yellow": "#EECC66",
    "light-red": "#EE99AA",
    "light-blue": "#6699CC",
    "dark-yellow": "#997700",
    "dark-red": "#994455",
    "dark-blue": "#004488",
    "black": "#000000",
}

def compare_contours(ax, xx, yy, grid, model1,model2,colors,m1_color,m2_color):
    ax = plt.gca()
    scales = np.linspace(-0.5,len(colors)-0.5, len(colors)+1)
    preds = model1(grid).detach().numpy()
    preds = np.argmax(preds, axis=1).reshape(xx.shape)
    ax.contour(xx, yy, preds, levels=scales, colors=m1_color)
    preds = model2(grid).detach().numpy()
    preds_student = np.argmax(preds, axis=1).reshape(xx.shape)
    ax.contour(xx, yy, preds_student, levels=scales,colors=m2_color)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.set_aspect('equal', adjustable='box')

def plot_contour_and_points(ax, xx, yy, grid, model, colors, contour_colors, X, y):
    preds = model(grid).detach().numpy()
    preds = np.argmax(preds, axis=1).reshape(xx.shape)
    scales = np.linspace(-0.5,len(colors)-0.5, len(colors)+1)
    ax.contourf(xx, yy, preds, levels=scales, colors=contour_colors, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=[colors[c.item()] for c in y],  edgecolors='white',marker='o',s=100)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.set_aspect('equal', adjustable='box')
    
def plot_correct_wrong(ax, X_transfer_test, y_transfer_test, model,name, accuracies):
    correct_samples = (model(X_transfer_test).argmax(dim=1) == y_transfer_test)
    ax.scatter(X_transfer_test[correct_samples, 0], X_transfer_test[correct_samples, 1], 
                edgecolors='green',marker='o',s=100, facecolors='none')
    ax.scatter(X_transfer_test[~correct_samples, 0], X_transfer_test[~correct_samples, 1], 
                marker='x',s=50, facecolors='red')
    ax.set_title(f'{name}\n${{{int(correct_samples.float().mean().item()*100)}}}$% test acc.')


def plot_summary(axes,models_3_label):
    xx, yy = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    

    ax = axes[0]

    plot_contour_and_points(ax, xx, yy, grid, 
                            models_3_label['teacher'], models_3_label['colors'], 
                            models_3_label['contour_colors'], 
                            models_3_label['X'], 
                            models_3_label['y'])
    ax.set_title('teacher\n100% train acc.')
    ax.set_aspect('equal', adjustable='box')
    
    ax = axes[1]

    plot_contour_and_points(ax, xx, yy, grid,
                            models_3_label['random'], models_3_label['colors'], 
                            models_3_label['contour_colors'], 
                            models_3_label['X_transfer_train'], 
                            models_3_label['y_transfer_train'])
    plot_correct_wrong(ax, models_3_label['X_transfer_test'],  models_3_label['y_transfer_test'], models_3_label['random'], name='independent',accuracies=models_3_label['random_accuracies'])


    ax = axes[2]

    plot_contour_and_points(ax, xx, yy, grid,
                            models_3_label['student'], models_3_label['colors'], 
                            models_3_label['contour_colors'], 
                            models_3_label['X_transfer_train'], 
                            models_3_label['y_transfer_train'])

    plot_correct_wrong(ax, models_3_label['X_transfer_test'],  models_3_label['y_transfer_test'], models_3_label['student'], name='student',accuracies=models_3_label['student_accuracies'])

    ax = axes[3]
    ax.set_title('decision boundaries\nteacher A vs. student B')

    compare_contours(ax, xx, yy, grid,
                        models_3_label['teacher'], models_3_label['student'],
                        models_3_label['colors'], 
                        'black', 
                        'tab:orange')
    
    