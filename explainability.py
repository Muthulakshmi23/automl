# src/explainability.py
import shap
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

def explain_tabular(xgb, tabular_features):
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(tabular_features)
    shap.summary_plot(shap_values, tabular_features)

def explain_image(resnet, image):
    target_layer = resnet.layer4[-1]
    cam = GradCAM(model=resnet, target_layers=[target_layer])
    input_tensor = torch.tensor(image).permute(0, 3, 1, 2).float()
    grayscale_cam = cam(input_tensor)[0]
    heatmap = show_cam_on_image(image, grayscale_cam)
    plt.imshow(heatmap)
    plt.show()
