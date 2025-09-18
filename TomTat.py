from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "./results_cluster_finetune"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.eval()

def summarize(text):
    inputs = tokenizer(
        text,
        max_length=1024, 
        truncation=True, 
        return_tensors="pt"
    )
    summary_ids = model.generate(
        inputs["input_ids"], 
        no_repeat_ngram_size=3,
        max_length=128, 
        num_beams=4, 
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    sample_text = """Phía Liên bang Nga chưa đưa ra bình luận tức thời về thông tin liên quan do Hải quân Ukraine công bố.

Theo báo The Kyiv Independent ngày 14/9, Sevastopol từ lâu đã là căn cứ của Hạm đội Biển Đen, trước cả khi Crimea bị sáp nhập vào Liên bang Nga năm 2014.

Các cuộc tấn công liên tiếp của Ukraine bằng phương tiện không người lái Hải quân, tên lửa và thiết bị bay không người lái tầm xa đã buộc Điện Kremlin phải cắt giảm sự hiện diện Hải quân trên bán đảo này.

Ukraine đã phá hủy nhiều tàu của Liên bang Nga, bao gồm tàu đổ bộ Caesar Kunikov, tàu tuần tra Sergei Kotov, tàu hộ vệ tên lửa Ivanovets, cùng nhiều tàu đổ bộ cao tốc khác.

Sự hiện diện ngày càng thu hẹp của Liên bang Nga tại Sevastopol diễn ra trong bối cảnh Ukraine gia tăng các cuộc tấn công bằng thiết bị bay không người lái và tên lửa nhằm vào những địa điểm khác của Hạm đội Biển Đen.

Cơ quan Tình báo Quân sự Ukraine (HUR) ngày 7/8 cho biết lực lượng nước này đã sử dụng thiết bị bay không người lái nhằm vào nhiều mục tiêu quân sự có giá trị cao tại Bán đảo Crimea, trong đó có một tàu đổ bộ cao tốc và ba trạm radar.

Theo tờ The Kyiv Independent, một trong những mục tiêu bị tấn công là tàu đổ bộ cao tốc Project 02510 BK-16, loại tàu chuyên phục vụ các hoạt động ven biển và vận chuyển binh sĩ. Cuộc tấn công được cho là đã khiến tàu này bốc cháy và hư hại.."""
    print("Tóm tắt:", summarize(sample_text))
