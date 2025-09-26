"""
Google Colab画像表示テスト - 最小限バージョン
画像アップロードと表示機能のみをテストします
"""

import io
import ipywidgets as widgets
from IPython.display import display, clear_output, Image as IPImage
from PIL import Image
import matplotlib.pyplot as plt

try:
    from google.colab import files
    import matplotlib
    matplotlib.use('Agg')
    plt.ioff()
    IN_COLAB = True
    print("✓ Google Colab環境を検出しました")
except ImportError:
    IN_COLAB = False
    print("✓ 通常のJupyter環境です")

class SimpleImageTest:
    def __init__(self):
        print("=== 画像表示テスト ===")
        self.current_image = None
        self.setup_ui()
    
    def setup_ui(self):
        """UIセットアップ"""
        self.upload_widget = widgets.FileUpload(
            accept='image/*',
            multiple=False,
            description='画像をアップロード'
        )
        self.upload_widget.observe(self.on_upload, names='value')
        
        self.output_widget = widgets.Output()
        
    def on_upload(self, change):
        """画像アップロード処理"""
        if not change['new']:
            return
            
        with self.output_widget:
            clear_output(wait=True)
            
        try:
            print("📁 ファイルを処理中...")
            uploaded_file = list(change['new'].values())[0]
            print(f"✓ ファイル名: {uploaded_file['metadata']['name']}")
            print(f"✓ ファイルサイズ: {len(uploaded_file['content'])} bytes")
            
            self.current_image = Image.open(io.BytesIO(uploaded_file['content']))
            print(f"✓ 画像読み込み成功")
            print(f"  サイズ: {self.current_image.size}")
            print(f"  モード: {self.current_image.mode}")
            
            self.test_display_methods()
            
        except Exception as e:
            print(f"❌ エラー: {e}")
            import traceback
            print(f"詳細: {traceback.format_exc()}")
    
    def test_display_methods(self):
        """複数の表示方法をテスト"""
        print("\n🧪 表示方法をテスト中...")
        
        try:
            print("\n📸 方法1: IPython.display.Image")
            img_bytes = io.BytesIO()
            self.current_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            display(IPImage(data=img_bytes.getvalue()))
            print("✓ IPython.display.Image表示成功")
        except Exception as e:
            print(f"❌ IPython.display.Image失敗: {e}")
        
        try:
            print("\n📊 方法2: matplotlib")
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(self.current_image)
            ax.set_title(f'Matplotlib表示テスト\nサイズ: {self.current_image.size}')
            ax.axis('off')
            plt.tight_layout()
            
            if IN_COLAB:
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                buf.seek(0)
                plt.close(fig)
                display(IPImage(data=buf.getvalue()))
                print("✓ matplotlib (Colab用) 表示成功")
            else:
                plt.show()
                print("✓ matplotlib (通常) 表示成功")
                
        except Exception as e:
            print(f"❌ matplotlib失敗: {e}")
            import traceback
            print(f"詳細: {traceback.format_exc()}")
        
        print(f"\n📋 画像情報:")
        print(f"  ファイル形式: {self.current_image.format}")
        print(f"  サイズ: {self.current_image.size}")
        print(f"  モード: {self.current_image.mode}")
        print(f"  環境: {'Google Colab' if IN_COLAB else 'Jupyter'}")
        
        print("\n✅ 表示テスト完了！上記のいずれかの方法で画像が表示されていることを確認してください。")
    
    def run(self):
        """テスト実行"""
        print("📋 手順:")
        print("1. 下のボタンから画像ファイルを選択してください")
        print("2. アップロード後、複数の表示方法をテストします")
        print("3. どの方法で画像が表示されるかを確認してください")
        print()
        
        display(self.upload_widget)
        display(self.output_widget)

def run_simple_test():
    """シンプルな画像表示テストを実行"""
    test = SimpleImageTest()
    test.run()

if __name__ == "__main__":
    run_simple_test()

print("\n" + "="*50)
print("Google Colab画像表示テスト準備完了")
print("run_simple_test() を実行してテストを開始してください")
print("="*50)
