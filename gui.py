import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from model import get_model
from utils import load_model, process_single_image

class MultiZooApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("MultiZoo Hayvan Sınıflandırıcı")
        self.geometry("900x600")
        self.minsize(900, 600)
        
        self.bg_color = "#f0f0f0"
        self.accent_color = "#4287f5"
        self.button_color = "#3366cc"
        self.text_color = "#333333"
        
        self.configure(bg=self.bg_color)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.create_widgets()
        self.load_model_btn.invoke()
    
    def create_widgets(self):
        top_frame = tk.Frame(self, bg=self.bg_color)
        top_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.load_model_btn = tk.Button(
            top_frame,
            text="Model Yükle",
            command=self.load_model_dialog,
            bg=self.button_color,
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            relief=tk.RAISED
        )
        self.load_model_btn.pack(side=tk.LEFT, padx=(0, 20))

        self.test_model_btn = tk.Button(
            top_frame,
            text="Test Modeli Seç",
            command=self.test_model,
            bg=self.button_color,
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            relief=tk.RAISED,
            state=tk.DISABLED
        )
        self.test_model_btn.pack(side=tk.LEFT, padx=(0, 20))
        
        self.select_image_btn = tk.Button(
            top_frame,
            text="Görüntü Seç",
            command=self.select_image,
            bg=self.button_color,
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            relief=tk.RAISED,
            state=tk.DISABLED
        )
        self.select_image_btn.pack(side=tk.LEFT, padx=(0, 20))
        
        self.model_info_label = tk.Label(
            top_frame,
            text="Model durumu: Yüklenmedi",
            bg=self.bg_color,
            fg=self.text_color,
            font=("Arial", 10)
        )
        self.model_info_label.pack(side=tk.LEFT, fill=tk.X)
        
        main_frame = tk.Frame(self, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        left_frame = tk.Frame(main_frame, bg=self.bg_color, width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.image_frame = tk.Frame(
            left_frame,
            bg="white",
            width=380,
            height=380,
            relief=tk.SUNKEN,
            bd=2
        )
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        self.image_frame.pack_propagate(False)
        
        self.image_label = tk.Label(
            self.image_frame,
            bg="white",
            text="Görüntü seçilmedi",
            font=("Arial", 12)
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        right_frame = tk.Frame(main_frame, bg=self.bg_color, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.prediction_title = tk.Label(
            right_frame,
            text="Tahmin Sonuçları",
            bg=self.bg_color,
            fg=self.accent_color,
            font=("Arial", 14, "bold")
        )
        self.prediction_title.pack(pady=(0, 10))
        
        self.prediction_frame = tk.Frame(
            right_frame,
            bg="white",
            relief=tk.SUNKEN,
            bd=2
        )
        self.prediction_frame.pack(fill=tk.BOTH, expand=True)
        
        self.prediction_label = tk.Label(
            self.prediction_frame,
            text="Henüz tahmin yapılmadı",
            bg="white",
            fg=self.text_color,
            font=("Arial", 12),
            justify=tk.CENTER
        )
        self.prediction_label.pack(fill=tk.BOTH, expand=True)
        
        self.predict_btn = tk.Button(
            right_frame,
            text="Tahmin Et",
            command=self.predict_image,
            bg=self.button_color,
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=10,
            relief=tk.RAISED,
            state=tk.DISABLED
        )
        self.predict_btn.pack(pady=20)
        
        self.status_bar = tk.Label(
            self,
            text="Hazır",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padx=10
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_model_dialog(self):
        model_path = filedialog.askopenfilename(
            title="Model Dosyasını Seç",
            filetypes=[("PyTorch Model", "*.pt"), ("Tüm Dosyalar", "*.*")]
        )
        
        if model_path:
            self.load_model_from_path(model_path)
    
    def load_model_from_path(self, model_path):
        try:
            self.model = get_model('vit', num_classes=90) 
            self.model, self.class_names = load_model(self.model, self.device, model_path)
            
            self.model_info_label.config(
                text=f"Model durumu: Yüklendi - {len(self.class_names)} sınıf"
            )
            self.select_image_btn.config(state=tk.NORMAL)
            self.test_model_btn.config(state=tk.NORMAL)
            self.status_bar.config(text=f"Model başarıyla yüklendi: {model_path}")
            
        except Exception as e:
            self.model_info_label.config(text=f"Model yükleme hatası!")
            self.status_bar.config(text=f"Hata: {str(e)}")
            print(f"Model yükleme hatası: {e}")
    
    def select_image(self):
        image_path = filedialog.askopenfilename(
            title="Görüntü Seç",
            filetypes=[
                ("Görüntü Dosyaları", "*.jpg *.jpeg *.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        
        if image_path:
            self.display_image(image_path)
            self.current_image_path = image_path
            self.predict_btn.config(state=tk.NORMAL)
            self.status_bar.config(text=f"Görüntü yüklendi: {image_path}")
    
    def display_image(self, image_path):
        try:
            pil_image = Image.open(image_path).convert("RGB")
            
            frame_width = self.image_frame.winfo_width()
            frame_height = self.image_frame.winfo_height()
            
            img_width, img_height = pil_image.size
            ratio = min(frame_width/img_width, frame_height/img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            tk_image = ImageTk.PhotoImage(pil_image)
            
            self.image_label.config(image=tk_image, text="")
            self.image_label.image = tk_image  # Referansı tut
            
        except Exception as e:
            self.image_label.config(text=f"Görüntü yükleme hatası!")
            self.status_bar.config(text=f"Hata: {str(e)}")
            print(f"Görüntü yükleme hatası: {e}")
    
    def predict_image(self):
        if not hasattr(self, 'current_image_path') or not self.model:
            self.status_bar.config(text="Lütfen önce bir görüntü seçin ve model yükleyin")
            return
        
        try:
            progress_window = tk.Toplevel(self)
            progress_window.title("İşleniyor")
            progress_window.geometry("300x100")
            progress_window.transient(self)
            progress_window.grab_set()
            
            progress_label = tk.Label(
                progress_window, 
                text="Görüntü işleniyor...", 
                font=("Arial", 10)
            )
            progress_label.pack(pady=10)
            
            progress_bar = ttk.Progressbar(
                progress_window, 
                orient=tk.HORIZONTAL, 
                length=250, 
                mode='indeterminate'
            )
            progress_bar.pack(pady=10)
            progress_bar.start()
            
            self.status_bar.config(text="Tahmin yapılıyor...")
            self.prediction_label.config(text="İşleniyor...")
            
            self.update()
            
            predicted_class, confidence = process_single_image(
                self.current_image_path,
                self.transform,
                self.model,
                self.device,
                self.class_names
            )
            
            result_text = f"Tahmin: {predicted_class}\n\nGüven: %{confidence*100:.2f}"
            self.prediction_label.config(text=result_text)
            self.status_bar.config(text=f"Tahmin tamamlandı: {predicted_class}")
            
            progress_window.destroy()
            
        except Exception as e:
            self.prediction_label.config(text=f"Tahmin hatası!")
            self.status_bar.config(text=f"Hata: {str(e)}")
            print(f"Tahmin hatası: {e}")
            
            if 'progress_window' in locals() and progress_window.winfo_exists():
                progress_window.destroy()

    def test_model(self):
        if not self.model:
            self.status_bar.config(text="Lütfen önce bir model yükleyin")
            return

        test_dir = filedialog.askdirectory(title="Test Klasörünü Seç")
        if not test_dir:
            return

        try:
            progress_window = tk.Toplevel(self)
            progress_window.title("Test İşlemi")
            progress_window.geometry("600x400")
            progress_window.transient(self)
            progress_window.grab_set()

            progress_label = tk.Label(
                progress_window,
                text="Test görüntüleri işleniyor...",
                font=("Arial", 12, "bold")
            )
            progress_label.pack(pady=20)

            progress_bar = ttk.Progressbar(
                progress_window,
                orient=tk.HORIZONTAL,
                length=550,
                mode='determinate'
            )
            progress_bar.pack(pady=20)

            results_text = tk.Text(progress_window, height=15, width=65, font=("Arial", 10))
            results_text.pack(pady=20)

            total_correct = 0
            total_images = 0

            for class_folder in os.listdir(test_dir):
                class_path = os.path.join(test_dir, class_folder)
                if not os.path.isdir(class_path):
                    continue

                for image_file in os.listdir(class_path):
                    if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue

                    image_path = os.path.join(class_path, image_file)
                    try:
                        prediction, confidence = process_single_image(
                            image_path,
                            self.transform,
                            self.model,
                            self.device,
                            self.class_names
                        )

                        total_images += 1
                        if prediction == class_folder:
                            total_correct += 1

                        results_text.insert(tk.END, 
                            f"{image_file}: Tahmin={prediction}, Doğru={class_folder}, "
                            f"Güven={confidence:.2f}\n")
                        results_text.see(tk.END)
                        progress_window.update()

                        progress_bar['value'] = (total_images / 
                            sum(len([f for f in os.listdir(os.path.join(test_dir, d)) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) 
                            for d in os.listdir(test_dir) 
                            if os.path.isdir(os.path.join(test_dir, d)))) * 100

                    except Exception as e:
                        print(f"Görüntü işleme hatası ({image_path}): {e}")

            accuracy = (total_correct / total_images) * 100 if total_images > 0 else 0
            results_text.insert(tk.END, f"\nToplam Doğruluk: {accuracy:.2f}%")
            results_text.see(tk.END)

        except Exception as e:
            self.status_bar.config(text=f"Test işlemi hatası: {str(e)}")
            print(f"Test işlemi hatası: {e}")

def main():
    app = MultiZooApp()
    app.mainloop()

if __name__ == "__main__":
    main() 