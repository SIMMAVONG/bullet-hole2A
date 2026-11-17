# python -m streamlit run bullet-hole2.py
import streamlit as st
import cv2
import numpy as np
from inference import get_model
import supervision as sv
import math

# ==================== ส่วนหัวโปรแกรมแบบจัดวางมุมซ้ายสุด (HEADER LAYOUT) ====================
# กำหนดชื่อไฟล์โลโก้ (ตรวจสอบให้แน่ใจว่าชื่อไฟล์ตรงกับที่คุณเซฟไว้ในโฟลเดอร์ เช่น 'V4.jpg')
logo_path = "V42.jpeg" 

# แบ่งพื้นที่ส่วนหัว: คอลัมน์ซ้ายสำหรับโลโก้ (ขนาดกะทัดรัด) และคอลัมน์ขวาสำหรับข้อความ
col_header_logo, col_header_text = st.columns([0.4, 3])

with col_header_logo:
    # แสดงรูปภาพโลโก้ชิดมุมซ้ายสุด
    st.image(logo_path, use_container_width=True)

with col_header_text:
    # แสดงชื่อวิทยาลัย (ภาษาลาว)
    st.markdown(
        """
        <h2 style='text-align: left; font-family: "Sarabun", sans-serif; margin-top: 10px; margin-bottom: 5px; color: #064e3b;'>
            ວິທະຍາຄານເຕັກນິກການທະຫານ
        </h2>
        """, 
        unsafe_allow_html=True
    )
st.markdown("---")
# =======================================================================================

# 1. ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title="Automatic Target Scoring", page_icon="🎯", layout="wide")
st.title("🎯 ລະບົບກວດຈັບ ແລະ ໃຫ້ຄະແນນຮູລູກກະສູນແບບອັດຕະໂນມັດ")
st.write("ອັບໂລດຮູບພາບແຜ່ນເປົ້າແບບ 6 ວົງຂອງເຈົ້າເຂົ້າມາເພື່ອເບີ່ງຜົນໄດ້ຮັບຂອງການກວດຈັບ ແລະ ໃຫ້ຄະແນນຜົນການຍີງ")

# 2. ฟังก์ชันโหลดโมเดล
@st.cache_resource
def load_model():
    # Model ID และ API Key 
    model_id = "bullet_hole_yolov8/2"
    api_key = "SFN5SsrZG3cAQBLdscLd" 
    model = get_model(model_id=model_id, api_key=api_key)
    return model

model = load_model()

# 3. ฟังก์ชันสำหรับหาจุดกึ่งกลางเป้าและรัศมี (เวอร์ชันอัปเดตแก้ปัญหามุมกล้องเอียง)
def get_target_center_and_radii(image):
    # แปลงภาพเป็นขาวดำและเบลอเพื่อลดสัญญาณรบกวน
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # แยกสีดำและขาว 
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    
    # ค้นหาเส้นขอบ
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
        
    valid_contours = [c for c in contours if cv2.contourArea(c) > 1000]
    if not valid_contours:
        return None, None
        
    # เลือกพื้นที่สีดำที่ใหญ่ที่สุด
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # หาจุดกึ่งกลางด้วยวิธี Moments (แม่นยำกว่าสำหรับรูปทรงที่บิดเบี้ยวตามมุมกล้อง)
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        center = (cX, cY)
    else:
        # สำรองกรณีคำนวณ Moment ไม่ได้
        (x, y), _ = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
    
    # หารัศมีเฉลี่ยจาก "พื้นที่ (Area)" ของสีดำแทนการตีวงกลมล้อมรอบจุดกว้างสุด
    area = cv2.contourArea(largest_contour)
    r_black = math.sqrt(area / math.pi)
    
    # ระยะห่างเฉลี่ยต่อ 1 วงคะแนน (สีดำมี 3 วงคือ 10, 9, 8 จึงหาร 3)
    spacing = r_black / 3.0
    
    # กำหนดรัศมีของแต่ละคะแนน (ปรับเพิ่มวงนอกสุดเล็กน้อยเพื่อชดเชยขอบกระดาษ)
    radii = {
        10: spacing * 2,
        9: spacing * 3,
        8: spacing * 4,
        7: spacing * 5,
        6: spacing * 6,
        5: spacing * 7.5  
    }
    return center, radii

# 4. สร้างปุ่มสำหรับอัปโหลดไฟล์ภาพ
uploaded_file = st.file_uploader("ເລືອກຟາຍຮູບພາບແຜ່ນເປົ້າທີ່ເຈົ້າຍີງ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, 1)

    st.write("⏳ ກໍາລັງປະມວນຜົນການກວດຈັບ ແລະ ໃຫ້ຄະແນນຜົນການຍີງເປົ້າ...")

    # สั่งรันโมเดลเพื่อตรวจจับรอยกระสุน
    results = model.infer(image)
    detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))
    
    # หาจุดกึ่งกลางเป้าและรัศมี
    target_center, ring_radii = get_target_center_and_radii(image)
    
    annotated_image = image.copy()
    total_score = 0
    scores_list = []

    if target_center and ring_radii:
        # วาดจุดกึ่งกลางเป้า (สีแดง) เพื่อให้เห็นชัดเจน
        cv2.circle(annotated_image, target_center, 5, (0, 0, 255), -1)
        
        # ตรวจสอบแต่ละรอยกระสุนที่พบ
        for bbox in detections.xyxy:
            x_min, y_min, x_max, y_max = bbox
            # หาจุดกึ่งกลางของรอยกระสุน
            bullet_x = (x_min + x_max) / 2
            bullet_y = (y_min + y_max) / 2
            bullet_center = (int(bullet_x), int(bullet_y))
            
            # คำนวณระยะห่างจากกึ่งกลางเป้า
            distance = math.sqrt((bullet_x - target_center[0])**2 + (bullet_y - target_center[1])**2)
            
            # ให้คะแนนตามระยะห่าง
            score = 0
            if distance <= ring_radii[10]: score = 10
            elif distance <= ring_radii[9]: score = 9
            elif distance <= ring_radii[8]: score = 8
            elif distance <= ring_radii[7]: score = 7
            elif distance <= ring_radii[6]: score = 6
            elif distance <= ring_radii[5]: score = 5
            
            # เก็บเฉพาะนัดที่ได้คะแนน (5-10) ถ้าหลุดขอบวง 5 ถือว่าได้ 0 ไม่เก็บลงลิสต์
            if score > 0:
                scores_list.append(score)
                total_score += score
            
            # วาดกรอบรอยกระสุนและใส่ตัวเลขคะแนนกำกับ
            cv2.rectangle(annotated_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 255), 2)
            cv2.putText(annotated_image, f"Score: {score}", (int(x_min), int(y_min) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.circle(annotated_image, bullet_center, 3, (0, 255, 0), -1)
            
    else:
        st.warning("⚠️ ລະບົບບໍ່ສາມາດຄົ້ນຫາຈຸດເຄີ່ງກາງຂອງແຜ່ນເປົ້າໄດ້ ກະລຸນາກວດສອບຄວາມຊັດເຈນຂອງຮູບພາບຕື່ມອິກ")

    # 5. แปลงระบบสีและแสดงผล
    original_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ຮູບພາບຕົ້ນສະບັບ")
        st.image(original_image_rgb, use_container_width=True)
        
    with col2:
        st.subheader("ຜົນໄດ້ຮັບໃນການກວດຈັບຂອງ Model CNN ແລະ ໃຫ້ຄະແນນຜົນການຍີງເປົ້າ")
        st.image(annotated_image_rgb, use_container_width=True)

    # 6. แสดงสรุปคะแนน
    st.markdown("---")
    st.subheader("📊 ສະຫຼູບຜົນການຍີງເປົ້າ")
    st.write(f"**ຈໍານວນນັດທີ່ເຂົ້າເປົ້າ (ວົງ 5-10):** {len(scores_list)} ນັດ")
    
    if len(scores_list) > 0:
        scores_str = " | ".join([f"ນັດທີ {i+1}: {s} ຄະແນນ" for i, s in enumerate(scores_list)])
        st.write(f"**ຄະແນນແຕ່ລະນັດ:** {scores_str}")
        st.success(f"🏆 **ຄະແນນລວມທັງໝົດ: {total_score} ຄະແນນ**")
    else:
        st.info("ບໍ່ພົບຮອຍລູກກະສູນໃນວົງຄະແນນ (ຄະແນນ: 0)")