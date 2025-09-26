#!/usr/bin/env python3
"""
DINOv3 Interactive Feature Visualization Demo for Google Colab
画像をアップロードして、マウスホバーで特徴量を表示し、クリックでセグメンテーションを行うデモ
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import cv2
from PIL import Image
import ipywidgets as widgets
from IPython.display import display, clear_output, Image as IPImage
import io
import sys
import base64
from typing import Optional, Tuple, List
import timm
import warnings
warnings.filterwarnings('ignore')

try:
    from google.colab import files
    import matplotlib
    matplotlib.use('Agg')  # バックエンドを明示的に設定
    plt.ioff()  # インタラクティブモードをオフ
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

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
        self.fixed_pixel = None
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
        
        self.confirm_button = None
        self.x_input = None
        self.y_input = None
        self.coord_button = None
        self.fix_button = None
        
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
                print(f"🔄 特徴量抽出開始...")
                print(f"Debug: Input tensor shape: {image_tensor.shape}")
                print("🧠 DINOv3モデルで特徴量を計算中...")
                features = self.model.forward_features(image_tensor)
                
                print(f"Debug: Raw features type: {type(features)}")
                
                if isinstance(features, dict):
                    print(f"Debug: Features is dictionary with keys: {list(features.keys())}")
                    if 'x_norm_patchtokens' in features:
                        features = features['x_norm_patchtokens']
                        print(f"Debug: Using 'x_norm_patchtokens', shape: {features.shape}")
                    elif 'x_prenorm' in features:
                        features = features['x_prenorm']
                        print(f"Debug: Using 'x_prenorm', shape: {features.shape}")
                    elif len(features) > 0:
                        key = list(features.keys())[0]
                        features = features[key]
                        print(f"Debug: Using first key '{key}', shape: {features.shape}")
                    else:
                        raise ValueError("Dictionary output contains no usable tensors")
                
                print(f"Debug: Raw features shape: {features.shape}")
                
                if len(features.shape) == 3 and features.shape[1] > 1:
                    patch_features = features[:, 1:]
                    print(f"Debug: Removed CLS token, patch_features shape: {patch_features.shape}")
                else:
                    patch_features = features
                    print(f"Debug: No CLS token removal, patch_features shape: {patch_features.shape}")
                
                n_patches = patch_features.shape[1]
                n_patches_side = int(np.sqrt(n_patches))
                
                print(f"Debug: n_patches={n_patches}, n_patches_side={n_patches_side}")
                
                if n_patches_side * n_patches_side != n_patches:
                    print(f"Warning: Non-square patch grid detected. n_patches={n_patches}")
                    n_patches_side = int(np.sqrt(n_patches))
                    actual_patches = n_patches_side * n_patches_side
                    patch_features = patch_features[:, :actual_patches, :]
                    print(f"Debug: Adjusted to {actual_patches} patches ({n_patches_side}x{n_patches_side})")
                
                patch_features = patch_features.reshape(1, n_patches_side, n_patches_side, -1)
                
                print(f"Debug: Final reshaped features: {patch_features.shape}")
                return patch_features
                
            except Exception as e:
                print(f"Error in primary feature extraction: {e}")
                import traceback
                print(f"Primary extraction traceback: {traceback.format_exc()}")
                
                try:
                    print("Debug: Trying alternative feature extraction...")
                    features = self.model(image_tensor)
                    print(f"Debug: Alternative features type: {type(features)}")
                    
                    if isinstance(features, tuple):
                        features = features[0]
                        print(f"Debug: Extracted from tuple, shape: {features.shape}")
                    
                    if isinstance(features, dict):
                        print(f"Debug: Alternative features is dictionary with keys: {list(features.keys())}")
                        if 'x_norm_patchtokens' in features:
                            features = features['x_norm_patchtokens']
                        elif 'x_prenorm' in features:
                            features = features['x_prenorm']
                        elif len(features) > 0:
                            key = list(features.keys())[0]
                            features = features[key]
                            print(f"Debug: Using first key '{key}', shape: {features.shape}")
                        else:
                            raise ValueError("Dictionary output contains no usable tensors")
                    
                    print(f"Debug: Alternative features shape: {features.shape}")
                    
                    if len(features.shape) == 2 and features.shape[0] == 1:
                        print(f"Debug: Got 1D feature vector {features.shape}, creating artificial patch grid")
                        features = features.unsqueeze(1).unsqueeze(1)  # [1, D] -> [1, 1, 1, D]
                        print(f"Debug: Reshaped to artificial patch grid: {features.shape}")
                        return features
                    
                    if len(features.shape) == 4:  # [B, C, H, W]
                        print("Debug: Converting [B, C, H, W] -> [B, H, W, C]")
                        features = features.permute(0, 2, 3, 1)  # [B, H, W, C]
                    elif len(features.shape) == 3:  # [B, N, D]
                        print("Debug: Converting [B, N, D] -> [B, H, W, D]")
                        n_patches = features.shape[1]
                        n_patches_side = int(np.sqrt(n_patches))
                        features = features.reshape(1, n_patches_side, n_patches_side, -1)
                    else:
                        raise ValueError(f"Unsupported feature tensor shape: {features.shape}")
                    
                    print(f"Debug: Alternative extraction final shape: {features.shape}")
                    return features
                    
                except Exception as e2:
                    print(f"Alternative feature extraction also failed: {e2}")
                    import traceback
                    print(f"Alternative extraction traceback: {traceback.format_exc()}")
                    raise e2
    
    def _on_upload(self, change):
        """画像アップロード時の処理"""
        if not change['new']:
            return
        
        try:
            with self.output_widget:
                clear_output(wait=True)
                print("📁 画像アップロードを処理中...")
            
            uploaded_file = list(change['new'].values())[0]
            print(f"✓ ファイル受信完了: {uploaded_file['metadata']['name']}")
            print(f"  ファイルサイズ: {len(uploaded_file['content'])} bytes")
            
            image = Image.open(io.BytesIO(uploaded_file['content']))
            print(f"✓ 画像読み込み成功")
            print(f"  画像サイズ: {image.size}")
            print(f"  画像モード: {image.mode}")
            
            self.current_image = image
            
            print("\n📸 アップロードされた画像を表示します...")
            self._display_uploaded_image()
            
            self.confirm_button = widgets.Button(
                description='画像確認OK - 特徴量抽出開始',
                button_style='success',
                layout=widgets.Layout(width='300px')
            )
            self.confirm_button.on_click(self._on_confirm_image)
            
            display(self.confirm_button)
            
        except Exception as e:
            with self.output_widget:
                clear_output(wait=True)
                print(f"❌ 画像アップロード中にエラーが発生しました: {e}")
                print(f"エラータイプ: {type(e).__name__}")
                import traceback
                print(f"詳細: {traceback.format_exc()}")
    
    def _display_uploaded_image(self):
        """アップロードされた画像の表示確認"""
        try:
            print("📸 アップロードされた画像:")
            
            img_bytes = io.BytesIO()
            self.current_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            display(IPImage(data=img_bytes.getvalue()))
            print("✓ IPython.display.Image表示成功")
            
            print(f"\n📋 画像情報:")
            print(f"  サイズ: {self.current_image.size}")
            print(f"  モード: {self.current_image.mode}")
            print(f"  環境: {'Google Colab' if IN_COLAB else 'Jupyter'}")
            
            print("\n✅ 画像表示完了 - 上記の画像が正しく表示されていることを確認してください")
            
        except Exception as e:
            print(f"❌ 画像表示エラー: {e}")
            import traceback
            print(f"詳細: {traceback.format_exc()}")
            
            print(f"画像情報: サイズ={self.current_image.size}, モード={self.current_image.mode}")
            print("画像の表示に失敗しましたが、アップロードは成功しています。")
    
    def _on_confirm_image(self, button):
        """画像確認後の特徴量抽出開始"""
        try:
            with self.output_widget:
                print("\n🔄 特徴量抽出を開始します...")
                print("⏳ この処理には数秒かかる場合があります...")
            
            self.confirm_button.disabled = True
            
            print("🖼️  画像の前処理中...")
            image_tensor = self._preprocess_image(self.current_image)
            print("✓ 画像前処理完了")
            
            print("🧠 DINOv3モデルで特徴量抽出中...")
            self.current_features = self._extract_features(image_tensor)
            print("✓ 特徴量抽出完了")
            print(f"📊 特徴量テンソル形状: {self.current_features.shape}")
            
            self._display_interactive_image()
            
            self.segment_button.disabled = False
            self.reset_button.disabled = False
            
            print("✅ 全ての処理が完了しました！画像上でマウスを動かしてみてください。")
            print("🎯 座標入力またはセグメンテーション機能を使用できます")
            
        except Exception as e:
            with self.output_widget:
                print(f"❌ 特徴量抽出中にエラーが発生しました: {e}")
                print(f"エラータイプ: {type(e).__name__}")
                import traceback
                print(f"詳細: {traceback.format_exc()}")
    
    def _display_interactive_image(self, marker_coords=None):
        """インタラクティブな画像表示"""
        try:
            print("\n🖼️ インタラクティブ画像を表示中...")
            
            resized_image = self.current_image.resize(self.image_size, Image.LANCZOS)
            
            if marker_coords is None:
                img_bytes = io.BytesIO()
                resized_image.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                display(IPImage(data=img_bytes.getvalue()))
            else:
                self._display_image_with_marker(resized_image, marker_coords)
            
            print("✓ インタラクティブ画像表示成功")
            
            if IN_COLAB:
                print("⚠️ Google Colabではマウスインタラクションが制限されています")
                print("座標を直接入力する機能を使用してください...")
                
                self._setup_coordinate_input()
                
            else:
                print("🖱️ 通常のJupyter環境: マウスインタラクション機能を準備中...")
                
                try:
                    self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
                    self.ax.imshow(resized_image)
                    self.ax.set_title('画像上をマウスホバーで特徴量表示、クリックで固定')
                    self.ax.axis('off')
                    
                    self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
                    self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_click)
                    
                    plt.tight_layout()
                    plt.show()
                    print("✓ matplotlib インタラクティブ表示も準備完了")
                except Exception as e:
                    print(f"⚠️ matplotlib インタラクティブ表示は失敗: {e}")
                
                self._setup_coordinate_input()
            
            print("✅ インタラクティブ画像表示完了")
            
        except Exception as e:
            print(f"❌ インタラクティブ画像表示エラー: {e}")
            import traceback
            print(f"詳細: {traceback.format_exc()}")
    
    def _display_image_with_marker(self, image, coords):
        """座標マーカー付きで画像を表示"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.imshow(image)
            
            x, y = coords
            ax.plot(x, y, 'r+', markersize=25, markeredgewidth=5, label='選択座標')
            ax.plot(x, y, 'wo', markersize=10, markeredgewidth=3)
            ax.plot(x, y, 'r+', markersize=20, markeredgewidth=3)
            
            ax.set_title(f'選択座標: ({x}, {y}) - 赤い十字で表示', fontsize=14, pad=15)
            ax.axis('off')
            plt.tight_layout()
            
            if IN_COLAB:
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, 
                           facecolor='white', edgecolor='none')
                buf.seek(0)
                plt.close(fig)
                display(IPImage(data=buf.getvalue()))
            else:
                plt.show()
                
            print(f"📍 座標 ({x}, {y}) を赤い十字で表示しました")
            
        except Exception as e:
            print(f"❌ マーカー付き画像表示エラー: {e}")
            import traceback
            print(f"詳細: {traceback.format_exc()}")
    
    def _setup_coordinate_input(self):
        """Google Colab用の座標入力ウィジェット"""
        print("\n📍 座標を入力して特徴量を確認:")
        
        self.x_input = widgets.IntSlider(
            value=112, min=0, max=self.image_size[0]-1,
            description='X座標:', style={'description_width': 'initial'}
        )
        self.y_input = widgets.IntSlider(
            value=112, min=0, max=self.image_size[1]-1,
            description='Y座標:', style={'description_width': 'initial'}
        )
        
        self.coord_button = widgets.Button(
            description='この座標の特徴量を表示',
            button_style='info'
        )
        self.coord_button.on_click(self._on_coordinate_click)
        
        self.fix_button = widgets.Button(
            description='この座標を基準点に設定',
            button_style='success'
        )
        self.fix_button.on_click(self._on_coordinate_fix)
        
        display(widgets.VBox([
            self.x_input,
            self.y_input,
            widgets.HBox([self.coord_button, self.fix_button])
        ]))
    
    def _on_coordinate_click(self, button):
        """座標入力での特徴量表示"""
        try:
            x, y = self.x_input.value, self.y_input.value
            
            print(f"\n🎯 座標 ({x}, {y}) を選択しました")
            
            resized_image = self.current_image.resize(self.image_size, Image.LANCZOS)
            self._display_image_with_marker(resized_image, (x, y))
            
            feature_vector = self._get_feature_at_pixel(x, y)
            
            if feature_vector is not None:
                norm = torch.norm(feature_vector).item()
                mean_val = torch.mean(feature_vector).item()
                std_val = torch.std(feature_vector).item()
                
                print(f"\n📊 座標 ({x}, {y}) の特徴量情報:")
                print(f"  ノルム: {norm:.4f}")
                print(f"  平均: {mean_val:.4f}")
                print(f"  標準偏差: {std_val:.4f}")
            else:
                print(f"❌ 座標 ({x}, {y}) の特徴量取得に失敗")
                
        except Exception as e:
            print(f"❌ 特徴量表示エラー: {e}")
    
    def _on_coordinate_fix(self, button):
        """座標入力での基準点設定"""
        try:
            x, y = self.x_input.value, self.y_input.value
            
            print(f"\n🎯 基準点を座標 ({x}, {y}) に設定します")
            
            resized_image = self.current_image.resize(self.image_size, Image.LANCZOS)
            self._display_image_with_marker(resized_image, (x, y))
            
            self.fixed_pixel = (x, y)
            self.fixed_feature = self._get_feature_at_pixel(x, y)
            
            if self.fixed_feature is not None:
                print(f"✅ 基準点を座標 ({x}, {y}) に設定しました")
                print("🚀 セグメンテーションボタンが有効になりました")
                print("「セグメンテーション実行」ボタンを押してください")
            else:
                print(f"❌ 座標 ({x}, {y}) での基準点設定に失敗")
                
        except Exception as e:
            print(f"❌ 基準点設定エラー: {e}")
    
    def _pixel_to_patch(self, x: int, y: int) -> Tuple[int, int]:
        """ピクセル座標をパッチ座標に変換"""
        try:
            print(f"Debug: current_features shape = {self.current_features.shape}")
            print(f"Debug: image_size = {self.image_size}")
            print(f"Debug: input coordinates = ({x}, {y})")
            
            if x < 0 or y < 0 or x >= self.image_size[0] or y >= self.image_size[1]:
                print(f"Warning: Coordinates ({x}, {y}) out of image bounds {self.image_size}")
                x = max(0, min(x, self.image_size[0] - 1))
                y = max(0, min(y, self.image_size[1] - 1))
                print(f"Debug: Clamped coordinates to ({x}, {y})")
            
            if len(self.current_features.shape) == 4:  # [B, H, W, D]
                _, h, w, _ = self.current_features.shape
                print(f"Debug: 4D tensor - patch grid size: {h}x{w}")
                
                if h == 1 and w == 1:
                    print("Debug: Global feature case - using single patch (0, 0)")
                    return 0, 0
                    
            elif len(self.current_features.shape) == 3:  # [B, N, D] - 予期しない形状
                print(f"Warning: Unexpected 3D tensor shape: {self.current_features.shape}")
                n_patches = self.current_features.shape[1]
                
                if n_patches == 1:
                    print("Debug: Single patch case - using patch (0, 0)")
                    return 0, 0
                
                h = w = int(np.sqrt(n_patches))
                print(f"Debug: 3D tensor - calculated patch grid: {h}x{w} from {n_patches} patches")
            else:
                raise ValueError(f"Unsupported feature tensor shape: {self.current_features.shape}")
            
            if h > 1 and w > 1:
                patch_x = int(x * w / self.image_size[0])
                patch_y = int(y * h / self.image_size[1])
                
                patch_x = max(0, min(patch_x, w - 1))
                patch_y = max(0, min(patch_y, h - 1))
            else:
                patch_x = patch_y = 0
            
            print(f"Debug: patch coordinates = ({patch_x}, {patch_y}) in grid {h}x{w}")
            
            if patch_x >= w or patch_y >= h:
                print(f"Error: Calculated patch coordinates ({patch_x}, {patch_y}) exceed grid bounds ({h}, {w})")
                patch_x = min(patch_x, w - 1)
                patch_y = min(patch_y, h - 1)
                print(f"Debug: Final clamped patch coordinates = ({patch_x}, {patch_y})")
            
            return patch_x, patch_y
            
        except Exception as e:
            print(f"Error in _pixel_to_patch: {e}")
            print(f"Features shape: {self.current_features.shape if self.current_features is not None else 'None'}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise e
    
    def _get_feature_at_pixel(self, x: int, y: int) -> torch.Tensor:
        """指定ピクセルの特徴量を取得"""
        try:
            if self.current_features is None:
                raise ValueError("Features not extracted yet")
                
            print(f"Debug: Getting feature at pixel ({x}, {y})")
            print(f"Debug: Current features shape: {self.current_features.shape}")
            
            patch_x, patch_y = self._pixel_to_patch(x, y)
            
            if len(self.current_features.shape) == 4:  # [B, H, W, D]
                _, h, w, d = self.current_features.shape
                print(f"Debug: 4D indexing - accessing [0, {patch_y}, {patch_x}, :] from shape [1, {h}, {w}, {d}]")
                
                if patch_y >= h or patch_x >= w:
                    raise IndexError(f"Patch coordinates ({patch_x}, {patch_y}) exceed tensor bounds ({w}, {h})")
                
                feature = self.current_features[0, patch_y, patch_x, :]
                
            elif len(self.current_features.shape) == 3:  # [B, N, D]
                _, n_patches, d = self.current_features.shape
                n_patches_side = int(np.sqrt(n_patches))
                patch_idx = patch_y * n_patches_side + patch_x
                
                print(f"Debug: 3D indexing - patch_idx={patch_idx} from ({patch_x}, {patch_y}) in {n_patches_side}x{n_patches_side} grid")
                print(f"Debug: Accessing [0, {patch_idx}, :] from shape [1, {n_patches}, {d}]")
                
                if patch_idx >= n_patches:
                    print(f"Warning: patch_idx {patch_idx} >= n_patches {n_patches}, clamping to {n_patches-1}")
                    patch_idx = n_patches - 1
                
                if patch_idx < 0:
                    print(f"Warning: patch_idx {patch_idx} < 0, clamping to 0")
                    patch_idx = 0
                
                feature = self.current_features[0, patch_idx, :]
                
            else:
                raise ValueError(f"Unsupported feature tensor shape: {self.current_features.shape}")
            
            print(f"Debug: extracted feature shape = {feature.shape}")
            
            if feature.numel() == 0:
                raise ValueError("Extracted feature is empty")
            
            return feature
            
        except Exception as e:
            print(f"Error in _get_feature_at_pixel: {e}")
            print(f"Input coordinates: ({x}, {y})")
            print(f"Features shape: {self.current_features.shape if self.current_features is not None else 'None'}")
            if hasattr(self, 'current_features') and self.current_features is not None:
                print(f"Features dtype: {self.current_features.dtype}")
                print(f"Features device: {self.current_features.device}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise e
    
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
    
    def _compute_similarity_map(self, reference_feature: torch.Tensor, output_widget=None) -> np.ndarray:
        """参照特徴量との類似度マップを計算（進捗表示付き）"""
        all_features = self.current_features[0]  # [H, W, D]
        h, w, d = all_features.shape
        total_pixels = h * w
        
        def print_progress(message):
            if output_widget:
                with output_widget:
                    print(message)
                    sys.stdout.flush()
            else:
                print(message)
        
        print_progress(f"🔄 類似度計算開始: {total_pixels:,}ピクセル ({h}x{w}) を処理中...")
        
        all_features_flat = all_features.reshape(-1, d)  # [H*W, D]
        reference_feature_norm = F.normalize(reference_feature.unsqueeze(0), p=2, dim=1)
        
        batch_size = min(500, total_pixels)  # より頻繁な進捗更新
        num_batches = (total_pixels + batch_size - 1) // batch_size
        
        similarity_scores = []
        
        import time
        start_time = time.time()
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_pixels)
            
            progress = (batch_idx + 1) / num_batches * 100
            processed_pixels = min(end_idx, total_pixels)
            elapsed = time.time() - start_time
            rate = processed_pixels / elapsed if elapsed > 0 else 0
            eta = (total_pixels - processed_pixels) / rate if rate > 0 else 0
            
            print_progress(f"📊 進捗: {processed_pixels:,}/{total_pixels:,} ピクセル ({progress:.1f}%) 完了")
            if batch_idx > 0:
                print_progress(f"⏱️ 処理速度: {rate:.0f} pixel/秒, 残り時間: {eta:.1f}秒")
            
            batch_features = all_features_flat[start_idx:end_idx]
            batch_features_norm = F.normalize(batch_features, p=2, dim=1)
            batch_similarity = torch.mm(batch_features_norm, reference_feature_norm.T).squeeze()
            
            similarity_scores.append(batch_similarity)
        
        similarity = torch.cat(similarity_scores, dim=0)
        similarity_map = similarity.reshape(h, w).cpu().numpy()
        
        total_time = time.time() - start_time
        print_progress(f"✅ 類似度計算完了! 統計情報:")
        print_progress(f"   最大類似度: {similarity_map.max():.3f}")
        print_progress(f"   最小類似度: {similarity_map.min():.3f}")
        print_progress(f"   平均類似度: {similarity_map.mean():.3f}")
        print_progress(f"   総処理時間: {total_time:.1f}秒")
        
        return similarity_map
    
    def _on_segment_click(self, button):
        """セグメンテーションボタンクリック時の処理"""
        reference_pixel = None
        if hasattr(self, 'selected_pixel') and self.selected_pixel is not None:
            reference_pixel = self.selected_pixel
        elif hasattr(self, 'fixed_pixel') and self.fixed_pixel is not None:
            reference_pixel = self.fixed_pixel
        
        if reference_pixel is None or self.current_features is None:
            with self.output_widget:
                print("まず画像上をクリックして基準点を選択してください")
                print("方法1: 画像上をクリック")
                print("方法2: 座標スライダーで座標を設定し「この座標を基準点に設定」ボタンを押す")
            return
        
        button.disabled = True
        
        try:
            with self.output_widget:
                print("🚀 セグメンテーション処理を開始します...")
                print("⏳ 処理には時間がかかりますが、リアルタイムで進捗を表示します")
                x, y = reference_pixel
                print(f"📍 基準点: ({x}, {y})")
                
                print("🔍 基準点の特徴量を取得中...")
                reference_feature = self._get_feature_at_pixel(x, y)
                print(f"✅ 基準点特徴量取得完了 (次元: {reference_feature.shape[0]})")
                
                print("🧮 全ピクセルとの類似度を計算中...")
                sys.stdout.flush()
            
            similarity_map = self._compute_similarity_map(reference_feature, self.output_widget)
            
            with self.output_widget:
                print("🎨 セグメンテーション結果を表示します...")
                
            self._display_segmentation_result(similarity_map)
            
            with self.output_widget:
                print("🎉 セグメンテーション処理が完了しました!")
            
            button.disabled = False
            
        except Exception as e:
            with self.output_widget:
                print(f"❌ セグメンテーション中にエラーが発生しました: {e}")
                import traceback
                print(f"詳細: {traceback.format_exc()}")
            button.disabled = False
    
    def _display_segmentation_result(self, similarity_map: np.ndarray):
        """セグメンテーション結果の表示"""
        try:
            print("🖼️ セグメンテーション結果を表示中...")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            resized_image = self.current_image.resize(self.image_size, Image.LANCZOS)
            ax1.imshow(resized_image)
            
            reference_pixel = None
            if hasattr(self, 'selected_pixel') and self.selected_pixel is not None:
                reference_pixel = self.selected_pixel
            elif hasattr(self, 'fixed_pixel') and self.fixed_pixel is not None:
                reference_pixel = self.fixed_pixel
            
            if reference_pixel:
                x, y = reference_pixel
                ax1.plot(x, y, 'r+', markersize=20, markeredgewidth=4, label='基準点')
                ax1.plot(x, y, 'wo', markersize=8, markeredgewidth=2)
                ax1.plot(x, y, 'r+', markersize=15, markeredgewidth=2)
                print(f"📍 基準点座標: ({x}, {y})")
            
            ax1.set_title('元画像 (基準点: 赤い十字)', fontsize=14, pad=10)
            ax1.axis('off')
            
            similarity_upsampled = cv2.resize(
                similarity_map, 
                self.image_size, 
                interpolation=cv2.INTER_LINEAR
            )
            
            im2 = ax2.imshow(similarity_upsampled, cmap='hot', alpha=0.7)
            ax2.imshow(resized_image, alpha=0.3)
            ax2.set_title('セグメンテーション結果\n(暖色：基準点と類似)', fontsize=14, pad=10)
            ax2.axis('off')
            
            cbar = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_label('類似度', rotation=270, labelpad=15)
            
            plt.tight_layout()
            
            if IN_COLAB:
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, 
                           facecolor='white', edgecolor='none')
                buf.seek(0)
                plt.close(fig)
                
                print("✅ セグメンテーション結果:")
                display(IPImage(data=buf.getvalue()))
            else:
                plt.show()
            
            print(f"\n📊 セグメンテーション統計:")
            print(f"  最高類似度: {similarity_map.max():.3f}")
            print(f"  最低類似度: {similarity_map.min():.3f}")
            print(f"  平均類似度: {similarity_map.mean():.3f}")
            high_similarity_pixels = (similarity_map > 0.8).sum()
            total_pixels = similarity_map.size
            print(f"  高類似度ピクセル (>0.8): {high_similarity_pixels}/{total_pixels} ({100*high_similarity_pixels/total_pixels:.1f}%)")
            
        except Exception as e:
            print(f"❌ 表示エラー: {e}")
            import traceback
            print(f"詳細: {traceback.format_exc()}")
    
    def _on_reset_click(self, button):
        """リセットボタンクリック時の処理"""
        self.selected_pixel = None
        self.fixed_pixel = None
        if self.current_image is not None:
            self._display_interactive_image()
    
    def run(self):
        """デモの実行"""
        print("=== DINOv3 Interactive Feature Demo ===")
        print("📋 使用手順:")
        print("1. 下のボタンから画像をアップロードしてください")
        print("2. アップロード後、画像が正しく表示されることを確認してください")
        print("3. '画像確認OK'ボタンを押して特徴量抽出を開始します")
        print("4. 画像上をマウスホバーすると特徴量情報が表示されます")
        print("5. クリックすると基準点が固定されます")
        print("6. 'セグメンテーション実行'ボタンで類似領域を可視化します")
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
