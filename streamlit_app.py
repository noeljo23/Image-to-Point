"""
RGB2Point Real-Time Demo
=========================
Upload an image → Generate 3D point cloud instantly

Run: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import torch
from torchvision import transforms
import time
import os

# ============================================================================
# Page Config
# ============================================================================

st.set_page_config(
    page_title="Image-to-Point Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
    }
    .success-banner {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Model Loading (Cached)
# ============================================================================


@st.cache_resource
def load_model():
    """Load Image-to-Point model (cached - only loads once)."""
    try:
        from model import PointCloudNet

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize with exact parameters from inference.py
        model = PointCloudNet(
            num_views=1,
            point_cloud_size=1024,
            num_heads=4,
            dim_feedforward=2048
        )

        # Find weights
        weight_paths = ["pc1024_three.pth"]
        weights_path = None
        for p in weight_paths:
            if os.path.exists(p):
                weights_path = p
                break

        if weights_path is None:
            return None, device, "Weights not found"

        checkpoint = torch.load(
            weights_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()

        return model, device, None

    except Exception as e:
        return None, "cpu", str(e)


def run_inference(model, device, image):
    """Run inference on PIL image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Model expects (batch, num_views, C, H, W) - add both dimensions
    img_tensor = transform(image).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        start = time.time()
        points = model(img_tensor)
        elapsed = time.time() - start

    # Reshape output to Nx3
    points_np = points.cpu().numpy().squeeze()
    if points_np.ndim == 1:
        points_np = points_np.reshape(-1, 3)

    return points_np, elapsed


# ============================================================================
# Visualization
# ============================================================================

def plot_3d(points, title="Point Cloud", colorscale='Viridis', size=2.5):
    """Create interactive 3D plot."""
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 2],
        z=points[:, 1],
        mode='markers',
        marker=dict(size=size, color=points[:, 1],
                    colorscale=colorscale, opacity=0.9)
    )])

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        scene=dict(
            xaxis_title='X', yaxis_title='Z', zaxis_title='Y',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=500
    )
    return fig


def comparison_chart():
    """F-Score comparison."""
    cats = ['Airplane', 'Car', 'Chair', 'Rifle', 'Sofa', 'Table', 'Average']
    ours = [0.581, 0.523, 0.544, 0.567, 0.481, 0.436, 0.505]
    sota = [0.589, 0.372, 0.309, 0.585, 0.224, 0.297, 0.343]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Image-to-Point (Ours)', x=cats, y=ours, marker_color='#667eea',
                         text=[f'{v:.3f}' for v in ours], textposition='outside'))
    fig.add_trace(go.Bar(name='PC² (Diffusion)', x=cats, y=sota, marker_color='#f59e0b',
                         text=[f'{v:.3f}' for v in sota], textposition='outside'))

    fig.update_layout(
        title='F-Score Comparison (Higher = Better)',
        barmode='group', height=400, yaxis_range=[0, 0.7],
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center')
    )
    return fig


# ============================================================================
# Load Model
# ============================================================================

model, device, error = load_model()

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown("## 🎯 Image-to-Point")
    st.markdown("*WACV 2025*")
    st.markdown("---")

    page = st.radio("", ["🏠 Overview", "🚀 Demo", "📊 Results"],
                    label_visibility="collapsed")


# ============================================================================
# Pages
# ============================================================================

if page == "🏠 Overview":

    st.markdown('<h1 class="main-header">🎯 Image-to-Point</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">3D Point Cloud Generation from Single RGB Images<br><em>Lee & Benes — WACV 2025</em></p>', unsafe_allow_html=True)

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    for col, (val, lbl) in zip([c1, c2, c3, c4], [
        ("15,133×", "Faster than Diffusion"),
        ("47.2%", "Higher F-Score"),
        ("2.3 GB", "VRAM Required"),
        ("0.15s", "Inference Time")
    ]):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    left, right = st.columns([3, 2])

    with left:
        st.markdown("""
        ## What is Image-to-Point?
        
        A **Transformer-based** model that generates 3D point clouds from **single RGB images**.
        
        ### Key Innovations
        
        - 🖼️ **Single Image** → No multi-view or depth sensors
        - ⚡ **Real-Time** → 15,000× faster than diffusion
        - 🎯 **High Quality** → Best F-score on ShapeNet & Pix3D
        - 💾 **Efficient** → Only 2.3GB VRAM
        
        ### Architecture
        
        1. **Vision Transformer (ViT)** → 768-dim features
        2. **CFI** → Multi-head attention (H=4)
        3. **GPM** → MLP to 3D coordinates
        """)

    with right:
        st.markdown("## Pipeline")
        st.code("""
📷 Input Image (224×224)
        ⬇️
🔍 ViT Encoder (768-dim)
        ⬇️
🧠 CFI Attention (H=4)
        ⬇️
📐 GPM Projection
        ⬇️
☁️ Point Cloud (1024×3)
        """)


elif page == "🚀 Demo":

    st.markdown("# 🚀 Real-Time Demo")
    st.markdown("Upload an image and generate a 3D point cloud instantly!")

    st.markdown("---")

    if model is None:
        st.error(f"❌ Model not loaded: {error}")
        st.info("Make sure `model.py` and `pc1024_three.pth` are in the same folder.")
        st.stop()

    left, right = st.columns(2)

    with left:
        st.markdown("### 📤 Upload Image")

        uploaded = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            help="For best results: clean background, single object, centered"
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Input Image", use_container_width=True)

            with st.expander("💡 Tips for Best Results"):
                st.markdown("""
                ✅ Clean, plain background (white is best)  
                ✅ Single object, centered  
                ✅ Good lighting  
                ❌ Avoid cluttered backgrounds  
                ❌ Avoid multiple objects
                """)
        else:
            st.markdown("""
            <div style="height: 300px; display: flex; align-items: center; justify-content: center; 
                        background: #f8fafc; border-radius: 12px; border: 2px dashed #cbd5e1;">
                <div style="text-align: center; color: #64748b;">
                    <div style="font-size: 3rem;">📷</div>
                    <div>Drag and drop an image</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown("### ☁️ Generated Point Cloud")

        if uploaded:
            if st.button("🚀 Generate Point Cloud", type="primary", use_container_width=True):
                with st.spinner("Generating..."):
                    points, elapsed = run_inference(model, device, image)

                st.session_state['points'] = points
                st.session_state['time'] = elapsed
                st.rerun()

            if 'points' in st.session_state:
                pts = st.session_state['points']
                t = st.session_state['time']

                m1, m2, m3 = st.columns(3)
                m1.metric("Points", f"{len(pts):,}")
                m2.metric("Time", f"{t:.3f}s")
                m3.metric("FPS", f"{1/t:.1f}")

                # Controls in expander to avoid overlap
                with st.expander("🎨 Visualization Options", expanded=False):
                    colorscale = st.selectbox(
                        "Color Scheme:", ['Viridis', 'Plasma', 'Turbo', 'Rainbow'])
                    pt_size = st.slider("Point Size:", 1.0, 5.0, 2.5)

                fig = plot_3d(pts, "", colorscale, pt_size)
                st.plotly_chart(fig, use_container_width=True)

                ply = f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
                ply += "\n".join([f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in pts])

                c1, c2 = st.columns(2)
                c1.download_button("📥 Download PLY", ply,
                                   "point_cloud.ply", use_container_width=True)
                if c2.button("🗑️ Clear", use_container_width=True):
                    del st.session_state['points']
                    st.rerun()
        else:
            st.markdown("""
            <div style="height: 400px; display: flex; align-items: center; justify-content: center; 
                        background: #f8fafc; border-radius: 12px; border: 2px dashed #cbd5e1;">
                <div style="text-align: center; color: #64748b;">
                    <div style="font-size: 3rem;">☁️</div>
                    <div>Upload an image to generate</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Multi-view
    if 'points' in st.session_state:
        st.markdown("---")
        st.markdown("### 🔄 Multiple Views")

        pts = st.session_state['points']
        v1, v2, v3 = st.columns(3)

        for col, (name, cam) in zip([v1, v2, v3], [
            ("Front", dict(eye=dict(x=0, y=2.5, z=0.3))),
            ("Side", dict(eye=dict(x=2.5, y=0, z=0.3))),
            ("Top", dict(eye=dict(x=0, y=0.1, z=2.5)))
        ]):
            with col:
                fig = go.Figure(data=[go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 2], z=pts[:, 1],
                    mode='markers',
                    marker=dict(size=2, color=pts[:, 1], colorscale='Viridis')
                )])
                fig.update_layout(
                    title=name, height=280,
                    scene=dict(camera=cam, aspectmode='cube'),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)


elif page == "📊 Results":

    st.markdown("# 📊 Benchmark Results")
    st.markdown("---")

    st.markdown("### F-Score Comparison on ShapeNet")
    st.plotly_chart(comparison_chart(), use_container_width=True)

    st.markdown('<div class="success-banner">Image-to-Point: 47.2% higher F-Score than diffusion SOTA!</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.markdown("### ⚡ Speed Comparison")

        fig = go.Figure(go.Bar(
            x=['Point-E', 'LION', 'Image-to-Point'],
            y=[2270, 45, 0.15],
            marker_color=['#ef4444', '#f59e0b', '#22c55e'],
            text=['37 min', '45 sec', '0.15 sec'],
            textposition='outside'
        ))
        fig.update_layout(yaxis_type='log',
                          yaxis_title='Seconds (log)', height=350)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### 📋 Metrics")
        st.markdown("""
        #### ShapeNet
        | Metric | Improvement |
        |--------|-------------|
        | Chamfer Distance | **-39.26%** |
        | Earth Mover's | **-26.95%** |
        | F-Score | **+47.2%** |
        
        #### Pix3D (Real-World)
        | Metric | Improvement |
        |--------|-------------|
        | Chamfer Distance | **-51.15%** |
        | Earth Mover's | **-36.17%** |
        """)

        st.success("🚀 **15,133× faster** than diffusion!")


# Footer
st.markdown("---")
st.markdown('<div style="text-align:center;color:#64748b;">Image-to-Point — Northeastern University</div>',
            unsafe_allow_html=True)
