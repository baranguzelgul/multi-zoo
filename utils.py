import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt

class MultiZooDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_mapping=None):
        
        self.root_dir = root_dir
        self.transform = transform
        
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.classes.sort() 

        if class_mapping is None:
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        else:
            self.class_to_idx = class_mapping

        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Hata: {img_path} dosyası yüklenemedi: {e}")
            return torch.zeros((3, 224, 224)), label

def create_data_loaders(train_dir, val_size=0.2, batch_size=32, image_size=224):

    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomAutocontrast(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = MultiZooDataset(root_dir=train_dir, transform=train_transforms)
    class_names = full_dataset.classes
    
    train_size = int((1 - val_size) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    val_dataset.dataset.transform = val_transforms
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, class_names

def evaluate_model(model, data_loader, device, criterion=None):

    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }
    
    if criterion is not None:
        loss = running_loss / len(data_loader.dataset)
        results['loss'] = loss
    
    return results

def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, train_precisions=None, val_precisions=None, 
                        train_recalls=None, val_recalls=None, train_f1s=None, val_f1s=None, save_path=None):

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Eğitim Kaybı')
    plt.plot(val_losses, label='Doğrulama Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()
    plt.title('Eğitim ve Doğrulama Kayıpları')
    if save_path:
        plt.savefig(save_path.replace('.png', '_loss.png'))
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label='Eğitim Doğruluğu')
    plt.plot(val_accs, label='Doğrulama Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()
    plt.title('Eğitim ve Doğrulama Doğrulukları')
    if save_path:
        plt.savefig(save_path.replace('.png', '_accuracy.png'))
    plt.show()
    
    if train_precisions is not None and val_precisions is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(train_precisions, label='Eğitim Kesinliği')
        plt.plot(val_precisions, label='Doğrulama Kesinliği')
        plt.xlabel('Epoch')
        plt.ylabel('Kesinlik')
        plt.legend()
        plt.title('Eğitim ve Doğrulama Kesinlikleri')
        if save_path:
            plt.savefig(save_path.replace('.png', '_precision.png'))
        plt.show()
    
    if train_recalls is not None and val_recalls is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(train_recalls, label='Eğitim Duyarlılığı')
        plt.plot(val_recalls, label='Doğrulama Duyarlılığı')
        plt.xlabel('Epoch')
        plt.ylabel('Duyarlılık')
        plt.legend()
        plt.title('Eğitim ve Doğrulama Duyarlılıkları')
        if save_path:
            plt.savefig(save_path.replace('.png', '_recall.png'))
        plt.show()
    
    if train_f1s is not None and val_f1s is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(train_f1s, label='Eğitim F1-skoru')
        plt.plot(val_f1s, label='Doğrulama F1-skoru')
        plt.xlabel('Epoch')
        plt.ylabel('F1-skor')
        plt.legend()
        plt.title('Eğitim ve Doğrulama F1-skorları')
        if save_path:
            plt.savefig(save_path.replace('.png', '_f1.png'))
        plt.show()

def plot_confusion_matrix(cm, class_names, save_path=None):

    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title('Karmaşıklık Matrisi')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def save_model(model, class_names, save_path='model.pt'):

    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, save_path)
    print(f"Model başarıyla kaydedildi: {save_path}")

def load_model(model, device, model_path='model.pt'):

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    class_names = checkpoint['class_names']
    model.to(device)
    model.eval()
    
    return model, class_names

def process_single_image(image_path, transform, model, device, class_names):

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted_idx.item()]
    confidence_value = confidence.item()
    
    return predicted_class, confidence_value 