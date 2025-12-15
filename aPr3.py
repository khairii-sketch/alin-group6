import streamlit as st
import numpy as np
from PIL import Image, ImageFilter

# PAGE CONFIG & LANGUAGE

st.set_page_config(page_title="Matrix Transform App", layout="centered")
lang = st.sidebar.selectbox("Language / Bahasa", ["English", "Indonesia"])
t = lambda en, id: en if lang == "English" else id
st.title(t("Matrix Transformation Webapp", "Aplikasi Web Transformasi Matriks"))

# MATRIX UTILITIES

def translation_matrix(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], float)

def scaling_matrix(sx, sy, center=(0,0)):
    cx, cy = center
    return translation_matrix(cx, cy) @ np.array([[sx,0,0],[0,sy,0],[0,0,1]],float) @ translation_matrix(-cx,-cy)

def rotation_matrix(angle_deg, center=(0,0)):
    teta = np.deg2rad(angle_deg)
    c, s = np.cos(teta), np.sin(teta)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]],float)
    cx, cy = center
    return translation_matrix(cx, cy) @ R @ translation_matrix(-cx,-cy)

def shearing_matrix(shx, shy):
    return np.array([[1,shx,0],[shy,1,0],[0,0,1]],float)

def reflection_matrix(axis, center=(0,0)):
    mats = {"x": np.array([[1,0,0],[0,-1,0],[0,0,1]],float),
            "y": np.array([[-1,0,0],[0,1,0],[0,0,1]],float),
            "y=x": np.array([[0,1,0],[1,0,0],[0,0,1]],float)}
    cx, cy = center
    return translation_matrix(cx, cy) @ mats[axis] @ translation_matrix(-cx,-cy)

# APPLY TRANSFORMATION

def apply_transform(img_arr, M):
    h, w = img_arr.shape[:2]
    inv = np.linalg.inv(M)
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    src = inv @ coords
    x_src, y_src = src[0]/src[2], src[1]/src[2]
    out = np.zeros_like(img_arr)

    for ch in range(img_arr.shape[2]):
        channel = img_arr[:,:,ch]
        x0, y0 = np.floor(x_src).astype(int), np.floor(y_src).astype(int)
        x1, y1 = np.clip(x0+1,0,w-1), np.clip(y0+1,0,h-1)
        x0, y0 = np.clip(x0,0,w-1), np.clip(y0,0,h-1)

        Ia, Ib, Ic, Id = channel[y0,x0], channel[y0,x1], channel[y1,x0], channel[y1,x1]
        wa, wb = (x1-x_src)*(y1-y_src), (x_src-x0)*(y1-y_src)
        wc, wd = (x1-x_src)*(y_src-y0), (x_src-x0)*(y_src-y0)
        out[:,:,ch] = (Ia*wa + Ib*wb + Ic*wc + Id*wd).reshape(h,w)
    return out.astype(np.uint8)

# IMAGE PROCESSING

def blur_image(img, level): return img.filter(ImageFilter.GaussianBlur(level))
def sharpen_image(img): return img.filter(ImageFilter.SHARPEN)
def remove_background_simple(img):
    gray = img.convert("L")
    arr = np.array(gray)
    mask = arr > 200
    rgb = np.array(img)
    rgb[mask] = [255,255,255]
    return Image.fromarray(rgb)

# SIDEBAR MENU

menu = st.sidebar.radio("Menu", ["Home & Teori", "Transformasi", "Image Processing", "Developer Team"])

# HOME & TEORI

if menu == "Home & Teori":
    st.subheader(t("Home", "Beranda"))
    st.write(t("Welcome to the Matrix Transformation Web Application.",
               "Selamat datang di Aplikasi Web Transformasi Matriks."))
    st.subheader(t("Theory", "Teori"))
    st.write("- Translation\n- Scaling\n- Rotation\n- Shearing\n- Reflection\n- Image Processing (Blur, Sharpen, Background Removal)")

# TRANSFORMASI

elif menu == "Transformasi":
    uploaded = st.file_uploader(t("Upload image", "Unggah gambar"), type=["jpg","jpeg","png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        arr = np.array(img)
        h, w = arr.shape[:2]
        center = (w/2, h/2)

        st.image(arr, caption=t("Original Image", "Gambar Asli"), use_column_width=True)
        tool = st.sidebar.selectbox(t("Choose transformation", "Pilih transformasi"),
                                    ["Translation","Scaling","Rotation","Shearing","Reflection"])

        if tool == "Translation":
            tx = st.sidebar.slider("Translate X", -200, 200, 0)
            ty = st.sidebar.slider("Translate Y", -200, 200, 0)
            M = translation_matrix(tx, ty)
        elif tool == "Scaling":
            sx = st.sidebar.slider("Scale X", 0.1, 3.0, 1.0)
            sy = st.sidebar.slider("Scale Y", 0.1, 3.0, 1.0)
            M = scaling_matrix(sx, sy, center)
        elif tool == "Rotation":
            angle = st.sidebar.slider("Angle", -180, 180, 0)
            M = rotation_matrix(angle, center)
        elif tool == "Shearing":
            shx = st.sidebar.slider("Shear X", -1.0, 1.0, 0.0)
            shy = st.sidebar.slider("Shear Y", -1.0, 1.0, 0.0)
            M = shearing_matrix(shx, shy)
        else:
            axis = st.sidebar.selectbox("Axis", ["x","y","y=x"])
            M = reflection_matrix(axis, center)

        st.image(apply_transform(arr, M), caption=t("Transformed Image", "Gambar Hasil"), use_column_width=True)
        st.code(M)


# IMAGE PROCESSING

elif menu == "Image Processing":
    uploaded = st.file_uploader(t("Upload image", "Unggah gambar"), type=["jpg","jpeg","png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption=t("Original Image", "Gambar Asli"), use_column_width=True)
        process = st.sidebar.selectbox(t("Choose process", "Pilih proses"), ["Blur", "Sharpen", "Background Removal"])
        result = blur_image(img, st.sidebar.slider(t("Blur level", "Tingkat blur"), 1, 10, 3)) if process=="Blur" else \
                 sharpen_image(img) if process=="Sharpen" else remove_background_simple(img)
        st.image(result, caption=t("Processed Image", "Hasil Pemrosesan"), use_column_width=True)

# DEVELOPER TEAM
else:
    st.subheader(t("Developer Team", "Tim Pengembang"))
    team = [{"name": "M. Gahril Firdaus [004202400118]", "role": "Matrix & Transformasi", "photo": "Anggota Sigma/M. Gahril Firdaus.jpg"},
            {"name": "Muhammad Dzakhwan Hafidz Khairi [004202400117]", "role": "Konvolusi & UI", "photo": "Anggota Sigma/Muhammad Dzakhwan Hafidz Khairi.jpg"},
            {"name": "Nailah Inayatul Chawari [004202400039]", "role": "Background Removal", "photo": "Anggota Sigma/Nailah Inayatul Chawari.jpg"},
            {"name": "Rachel Sharma Clarashita [004202400124]", "role": "Testing & Dokumentasi", "photo": "Anggota Sigma/Rachel Sharma Clarashita.jpg"}]

    for member in team:
        st.markdown(f"**{member['name']}**")
        st.markdown(f"Peran: {member['role']}")
        try:
            img = Image.open(member["photo"])
            st.image(img, caption=f"Foto {member['name']}", width=200)
        except Exception:
            st.warning("Foto tidak ditemukan.")
        st.markdown("---")