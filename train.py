import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import time
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from utils import create_data_loaders, evaluate_model, plot_learning_curves, plot_confusion_matrix, save_model
from model import get_model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=30, early_stopping_patience=5):

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_precisions = []
    val_precisions = []
    train_recalls = []
    val_recalls = []
    train_f1s = []
    val_f1s = []
    
    best_val_acc = 0.0
    best_model_wts = model.state_dict()
    no_improve_epochs = 0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        all_train_preds = []
        all_train_labels = []
        
        loop = tqdm(train_loader, desc=f"Eğitim")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

            loop.set_postfix(loss=loss.item())
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects / len(train_loader.dataset)
        epoch_train_precision = precision_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
        epoch_train_recall = recall_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
        epoch_train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
        
        val_results = evaluate_model(model, val_loader, device, criterion)
        epoch_val_loss = val_results['loss']
        epoch_val_acc = val_results['accuracy']
        epoch_val_precision = val_results['precision']
        epoch_val_recall = val_results['recall']
        epoch_val_f1 = val_results['f1']

        scheduler.step(epoch_val_loss)
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        train_precisions.append(epoch_train_precision)
        val_precisions.append(epoch_val_precision)
        train_recalls.append(epoch_train_recall)
        val_recalls.append(epoch_val_recall)
        train_f1s.append(epoch_train_f1)
        val_f1s.append(epoch_val_f1)
        
        print(f"Eğitim Kaybı: {epoch_train_loss:.4f}, Doğrulama Kaybı: {epoch_val_loss:.4f}")
        print(f"Eğitim Doğruluğu: {epoch_train_acc:.4f}, Doğrulama Doğruluğu: {epoch_val_acc:.4f}")
        print(f"Eğitim Kesinliği: {epoch_train_precision:.4f}, Doğrulama Kesinliği: {epoch_val_precision:.4f}")
        print(f"Eğitim Duyarlılığı: {epoch_train_recall:.4f}, Doğrulama Duyarlılığı: {epoch_val_recall:.4f}")
        print(f"Eğitim F1-skoru: {epoch_train_f1:.4f}, Doğrulama F1-skoru: {epoch_val_f1:.4f}")
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_wts = model.state_dict().copy()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            
        if no_improve_epochs >= early_stopping_patience:
            print(f"Erken durdurma: Doğrulama doğruluğu {early_stopping_patience} epoch boyunca iyileşmedi.")
            break
    
    model.load_state_dict(best_model_wts)
    
    return train_losses, val_losses, train_accs, val_accs, train_precisions, val_precisions, \
           train_recalls, val_recalls, train_f1s, val_f1s, model

def main():
    parser = argparse.ArgumentParser(description='MultiZoo Veri Seti için Transformer tabanlı model eğitimi')
    parser.add_argument('--train_dir', type=str, default='train', help='Eğitim verilerinin dizini')
    parser.add_argument('--model_type', type=str, default='vit', choices=['vit', 'swin', 'deit'], help='Kullanılacak model türü')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch boyutu')
    parser.add_argument('--epochs', type=int, default=15, help='Eğitim epoch sayısı')
    parser.add_argument('--lr', type=float, default=0.00005, help='Öğrenme oranı')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Ağırlık azaltma')
    parser.add_argument('--image_size', type=int, default=224, help='Görüntü boyutu')
    parser.add_argument('--val_size', type=float, default=0.2, help='Doğrulama seti oranı')
    parser.add_argument('--early_stopping', type=int, default=10, help='Erken durdurma sabır sayısı')
    parser.add_argument('--save_dir', type=str, default='results', help='Sonuçların kaydedileceği dizin')
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    train_loader, val_loader, class_names = create_data_loaders(
        args.train_dir, 
        val_size=args.val_size, 
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    num_classes = len(class_names)
    print(f"Sınıf sayısı: {num_classes}")
    
    model = get_model(args.model_type, num_classes, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    print("Model eğitimi başlıyor...")
    start_time = time.time()
    
    train_losses, val_losses, train_accs, val_accs, train_precisions, val_precisions, \
    train_recalls, val_recalls, train_f1s, val_f1s, best_model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping
    )
    
    training_time = time.time() - start_time
    print(f"Eğitim tamamlandı. Geçen süre: {training_time:.2f} saniye")
    
    plot_learning_curves(
        train_losses, 
        val_losses, 
        train_accs, 
        val_accs,
        train_precisions,
        val_precisions,
        train_recalls,
        val_recalls,
        train_f1s,
        val_f1s,
        save_path=os.path.join(args.save_dir, 'learning_curves.png')
    )
    
    print("Doğrulama setinde son değerlendirme yapılıyor...")
    val_results = evaluate_model(best_model, val_loader, device)
    
    print("\nDoğrulama Sonuçları:")
    print(f"Doğruluk (Accuracy): {val_results['accuracy']:.4f}")
    print(f"Kesinlik (Precision): {val_results['precision']:.4f}")
    print(f"Duyarlılık (Recall): {val_results['recall']:.4f}")
    print(f"F1-Skor: {val_results['f1']:.4f}")
    
    plot_confusion_matrix(
        val_results['confusion_matrix'], 
        class_names, 
        save_path=os.path.join(args.save_dir, 'confusion_matrix.png')
    )
    
    save_model(best_model, class_names, save_path=os.path.join(args.save_dir, 'model.pt'))
    
    with open(os.path.join(args.save_dir, 'results.txt'), 'w') as f:
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Eğitim süresi: {training_time:.2f} saniye\n\n")
        f.write("Doğrulama Sonuçları:\n")
        f.write(f"Doğruluk (Accuracy): {val_results['accuracy']:.4f}\n")
        f.write(f"Kesinlik (Precision): {val_results['precision']:.4f}\n")
        f.write(f"Duyarlılık (Recall): {val_results['recall']:.4f}\n")
        f.write(f"F1-Skor: {val_results['f1']:.4f}\n")
    
if __name__ == "__main__":
    main() 