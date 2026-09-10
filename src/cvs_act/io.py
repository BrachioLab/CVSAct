"""Image encoding and PDF save helpers."""

import base64
import io

import matplotlib.pyplot as plt
from PIL import Image


def img_to_data_url(path):
    """Convert an image file to a base64 data URL."""
    img = Image.open(path)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def save_and_show(fig, pdf):
    """Save figure to PdfPages, display in notebook, then close."""
    if fig is None:
        return
    pdf.savefig(fig)
    plt.show()
    plt.close(fig)


def save_only(fig, pdf):
    """Save figure to PdfPages (no notebook display), then close."""
    if fig is None:
        return
    pdf.savefig(fig)
    plt.close(fig)
