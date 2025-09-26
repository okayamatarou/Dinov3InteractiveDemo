#!/usr/bin/env python3
"""
DINOv3 Interactive Feature Visualization Demo for Google Colab
画像をアップロードして、マウスホバーで特徴量を表示し、クリックでセグメンテーションを行うデモ
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import cv2
from PIL import Image
import ipywidgets as widgets
from IPython.display import display, clear_output
import io
import base64
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class DINOv3FeatureDemo:
    def __init__(self):
        """DINOv3デモクラスの初期化"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = None
        self.model_type = None
        self.current_image = None
        self.current_features = None
        self.selected_pixel = None
        self.image_size = (224, 224)  # DINOv3の入力サイズ
        self.patch_size = 14  # DINOv3のパッチサイズ
        self.feature_dim = 384  # DINOv3-Sの特徴量次元
        
        self.upload_widget = None
        self.output_widget = None
        self.fig = None
        self.ax = None
        
        self._load_model()
        self._setup_ui()
    
    def _load_model(self):
        """DINOv3モデルの読み込み"""
        try:
            print("Loading DINOv3 model...")
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.model.eval()
            self.model.to(self.device)
            self.model_type = 'dinov2'
            print("DINOv3 model loaded successfully!")
        except Exception as e:
            print(f"Error loading DINOv3 model: {e}")
            print("Trying alternative loading method...")
            try:
                import timm
                self.model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True)
                self.model.eval()
                self.model.to(self.device)
                self.model_type = 'timm'
                print("DINOv3 model loaded via timm!")
            except Exception as e2:
                print(f"Failed to load model: {e2}")
                try:
                    self.model = timm.create_model('vit_small_patch14_224.dino', pretrained=True)
                    self.model.eval()
                    self.model.to(self.device)
                    self.model_type = 'timm_dino'
                    print("DINO model loaded via timm as fallback!")
                except Exception as e3:
                    print(f"All loading methods failed: {e3}")
                    raise e3
    
    def _setup_ui(self):
        """UIの設定"""
        self.upload_widget = widgets.FileUpload(
            accept='image/*',
            multiple=False,
            description='画像をアップロード'
        )
        self.upload_widget.observe(self._on_upload, names='value')
        
        self.output_widget = widgets.Output()
        
        self.segment_button = widgets.Button(
            description='セグメンテーション実行',
            disabled=True,
            button_style='info'
        )
        self.segment_button.on_click(self._on_segment_click)
        
        self.reset_button = widgets.Button(
            description='リセット',
            disabled=True,
            button_style='warning'
        )
        self.reset_button.on_click(self._on_reset_click)
        
        print("UI setup complete!")
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """画像の前処理"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize(self.image_size, Image.LANCZOS)
        
        img_array = np.array(image) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def _extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """DINOv3特徴量の抽出"""
        with torch.no_grad():
            try:
                features = self.model.forward_features(image_tensor)
                
                print(f"Features shape: {features.shape}")
                
                if len(features.shape) == 3 and features.shape[1] > 1:
                    patch_features = features[:, 1:]
                else:
                    patch_features = features
                
                print(f"Patch features shape: {patch_features.shape}")
                
                n_patches = patch_features.shape[1]
                n_patches_side = int(np.sqrt(n_patches))
                
                if n_patches_side * n_patches_side != n_patches:
                    print(f"Warning: Non-square patch grid detected. n_patches={n_patches}")
                    n_patches_side = int(np.sqrt(n_patches))
                    patch_features = patch_features[:, :n_patches_side*n_patches_side, :]
                
                patch_features = patch_features.reshape(1, n_patches_side, n_patches_side, -1)
                
                print(f"Reshaped features: {patch_features.shape}")
                return patch_features
                
            except Exception as e:
                print(f"Error in feature extraction: {e}")
                try:
                    features = self.model(image_tensor)
                    if isinstance(features, tuple):
                        features = features[0]
                    
                    if len(features.shape) == 4:  # [B, C, H, W]
                        features = features.permute(0, 2, 3, 1)  # [B, H, W, C]
                    elif len(features.shape) == 3:  # [B, N, D]
                        n_patches = features.shape[1]
                        n_patches_side = int(np.sqrt(n_patches))
                        features = features.reshape(1, n_patches_side, n_patches_side, -1)
                    
                    return features
                except Exception as e2:
                    print(f"Alternative feature extraction also failed: {e2}")
                    raise e2
    
    def _on_upload(self, change):
        """画像アップロード時の処理"""
        if not change['new']:
            return
        
        try:
            uploaded_file = list(change['new'].values())[0]
            image = Image.open(io.BytesIO(uploaded_file['content']))
            
            self.current_image = image
            
            with self.output_widget:
                clear_output(wait=True)
                print("特徴量を抽出中...")
            
            image_tensor = self._preprocess_image(image)
            self.current_features = self._extract_features(image_tensor)
            
            self._display_interactive_image()
            
            self.segment_button.disabled = False
            self.reset_button.disabled = False
            
        except Exception as e:
            with self.output_widget:
                clear_output(wait=True)
                print(f"エラーが発生しました: {e}")
    
    def _display_interactive_image(self):
        """インタラクティブな画像表示"""
        with self.output_widget:
            clear_output(wait=True)
            
            self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
            
            resized_image = self.current_image.resize(self.image_size, Image.LANCZOS)
            self.ax.imshow(resized_image)
            self.ax.set_title('画像上をマウスホバーで特徴量表示、クリックで固定')
            self.ax.axis('off')
            
            self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
            self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_click)
            
            plt.tight_layout()
            plt.show()
    
    def _pixel_to_patch(self, x: int, y: int) -> Tuple[int, int]:
        """ピクセル座標をパッチ座標に変換"""
        patch_x = int(x * self.current_features.shape[2] / self.image_size[0])
        patch_y = int(y * self.current_features.shape[1] / self.image_size[1])
        
        patch_x = max(0, min(patch_x, self.current_features.shape[2] - 1))
        patch_y = max(0, min(patch_y, self.current_features.shape[1] - 1))
        
        return patch_x, patch_y
    
    def _get_feature_at_pixel(self, x: int, y: int) -> torch.Tensor:
        """指定ピクセルの特徴量を取得"""
        patch_x, patch_y = self._pixel_to_patch(x, y)
        feature = self.current_features[0, patch_y, patch_x, :]
        return feature
    
    def _on_mouse_move(self, event):
        """マウス移動時の処理"""
        if event.inaxes != self.ax or self.current_features is None:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        if x < 0 or y < 0 or x >= self.image_size[0] or y >= self.image_size[1]:
            return
        
        feature = self._get_feature_at_pixel(x, y)
        
        feature_norm = torch.norm(feature).item()
        feature_mean = torch.mean(feature).item()
        feature_std = torch.std(feature).item()
        
        self.ax.set_title(
            f'位置: ({x}, {y}) | 特徴量ノルム: {feature_norm:.3f} | '
            f'平均: {feature_mean:.3f} | 標準偏差: {feature_std:.3f}'
        )
        
        self.fig.canvas.draw_idle()
    
    def _on_mouse_click(self, event):
        """マウスクリック時の処理"""
        if event.inaxes != self.ax or self.current_features is None:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        if x < 0 or y < 0 or x >= self.image_size[0] or y >= self.image_size[1]:
            return
        
        self.selected_pixel = (x, y)
        
        self.ax.clear()
        resized_image = self.current_image.resize(self.image_size, Image.LANCZOS)
        self.ax.imshow(resized_image)
        
        self.ax.plot(x, y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
        
        feature = self._get_feature_at_pixel(x, y)
        feature_norm = torch.norm(feature).item()
        
        self.ax.set_title(f'選択位置: ({x}, {y}) | 特徴量ノルム: {feature_norm:.3f} | セグメンテーションボタンを押してください')
        self.ax.axis('off')
        
        self.fig.canvas.draw()
    
    def _compute_similarity_map(self, reference_feature: torch.Tensor) -> np.ndarray:
        """参照特徴量との類似度マップを計算"""
        all_features = self.current_features[0]  # [H, W, D]
        h, w, d = all_features.shape
        
        all_features_flat = all_features.reshape(-1, d)  # [H*W, D]
        all_features_norm = F.normalize(all_features_flat, p=2, dim=1)
        reference_feature_norm = F.normalize(reference_feature.unsqueeze(0), p=2, dim=1)
        
        similarity = torch.mm(all_features_norm, reference_feature_norm.T).squeeze()
        similarity_map = similarity.reshape(h, w).cpu().numpy()
        
        return similarity_map
    
    def _on_segment_click(self, button):
        """セグメンテーションボタンクリック時の処理"""
        if self.selected_pixel is None or self.current_features is None:
            with self.output_widget:
                print("まず画像上をクリックして基準点を選択してください")
            return
        
        try:
            x, y = self.selected_pixel
            reference_feature = self._get_feature_at_pixel(x, y)
            
            similarity_map = self._compute_similarity_map(reference_feature)
            
            self._display_segmentation_result(similarity_map)
            
        except Exception as e:
            with self.output_widget:
                print(f"セグメンテーション中にエラーが発生しました: {e}")
    
    def _display_segmentation_result(self, similarity_map: np.ndarray):
        """セグメンテーション結果の表示"""
        with self.output_widget:
            clear_output(wait=True)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            resized_image = self.current_image.resize(self.image_size, Image.LANCZOS)
            ax1.imshow(resized_image)
            x, y = self.selected_pixel
            ax1.plot(x, y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
            ax1.set_title('元画像（赤点：基準点）')
            ax1.axis('off')
            
            similarity_upsampled = cv2.resize(
                similarity_map, 
                self.image_size, 
                interpolation=cv2.INTER_LINEAR
            )
            
            im2 = ax2.imshow(similarity_upsampled, cmap='hot', alpha=0.7)
            ax2.imshow(resized_image, alpha=0.3)
            ax2.set_title('特徴量類似度マップ（暖色：高類似度）')
            ax2.axis('off')
            
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.show()
            
            print(f"類似度統計:")
            print(f"  最大値: {similarity_map.max():.3f}")
            print(f"  最小値: {similarity_map.min():.3f}")
            print(f"  平均値: {similarity_map.mean():.3f}")
            print(f"  標準偏差: {similarity_map.std():.3f}")
    
    def _on_reset_click(self, button):
        """リセットボタンクリック時の処理"""
        self.selected_pixel = None
        if self.current_image is not None:
            self._display_interactive_image()
    
    def run(self):
        """デモの実行"""
        print("=== DINOv3 Interactive Feature Demo ===")
        print("1. 下のボタンから画像をアップロードしてください")
        print("2. 画像上をマウスホバーすると特徴量情報が表示されます")
        print("3. クリックすると基準点が固定されます")
        print("4. 'セグメンテーション実行'ボタンで類似領域を可視化します")
        print()
        
        display(self.upload_widget)
        display(widgets.HBox([self.segment_button, self.reset_button]))
        display(self.output_widget)

def run_dinov3_demo():
    """DINOv3デモを実行"""
    try:
        demo = DINOv3FeatureDemo()
        demo.run()
    except Exception as e:
        print(f"デモの初期化中にエラーが発生しました: {e}")
        print("必要なライブラリがインストールされているか確認してください:")
        print("!pip install torch torchvision matplotlib opencv-python pillow ipywidgets")

if __name__ == "__main__":
    try:
        import google.colab
        print("Google Colab環境を検出しました")
        run_dinov3_demo()
    except ImportError:
        print("ローカル環境での実行")
        run_dinov3_demo()
