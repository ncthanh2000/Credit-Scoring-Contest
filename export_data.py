#Basic Libraries
import pandas as pd
# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Import Data
train = pd.read_excel(r'Final Data/Train - FillNa Data.xlsx')
test = pd.read_excel(r'Final Data/Test - FillNa Data.xlsx')

# Setup Train Test Data
X = train.drop(columns = ['label', 'id'])
y = train['label']

X_train, X_train_cv, y_train, y_train_cv = train_test_split(X,y, test_size=0.05, random_state=42)

#Standardise before oversampling for SMOTE
def macv_dummy(_df):
    k1 = 'BS'  # Bác Sĩ
    k2 = 'CN'  # Công nhân
    k3 = 'CB'  # Cán Bộ
    k4 = 'CT'  # Chủ tịch
    k5 = 'CV'  # Chuyên viên
    k7 = 'DD'  # Điều dưỡng
    k8 = 'DS'  # Dược Sĩ
    k9 = 'GD'  # Giám đốc, PGD
    k10 = 'GS'  # Giám sát
    k11 = 'GV'  # Giáo viên
    k12 = 'NV'  # Nhân viên
    k13 = 'TP'  # Tổ phó
    k14 = 'TT'  # Tổ trưởng
    k15 = 'TL'  # Trợ Lý
    k16 = 'Tho'  # Thợ
    k17 = 'KT'  # Kĩ thuật
    k18 = 'LX'  # Lái các loại xe/máy
    k19 = 'YS'  # Y sĩ/tá
    k20 = 'TPK'  # Trưởng phòng/khoa
    k21 = 'PPK'  # Phó phòng/khoa
    k22 = 'PGD'  # Ph
    k23 = 'KS'  # Ki su/KTV
    k24 = 'KT'  # Kiem dinh/Ke Toan
    k25 = 'LD'  # Lao dong
    k26 = 'Fill' # NaN Job

    v1 = ['Bác sĩ', 'bác sĩ', 'Bác sỹ']
    v2 = ["004672 Công nhân", "21996CÔNG NHÂN", "cn", "Cn", "CN", "CN - UMDE9", "CN Bảo trì", "CN Bảo vệ",
          "CN bộ phận may", "CN CắT BIÊN", "CN cắt nhiệt", "CN cắt phá  cắt gọt", "CN Chà nhám",
          "CN chăm sóc và cạo mủ", "cn chặt liệu", "cn cơ khí", "CN CọC KHOAN NHồI", "CN CUỐN DA", "CN Dệt",
          "CN Đóng Gói", "cn đóng kiện trong dây chuyền may", "CN Đột dập", "cn gạch", "CN giặt quần bò",
          "CN Giặt quần bò", "CN Hà Nội - V1 - NV Nhóm Chuyên doanh Laptop", "CN Kéo khô", "CN Kho", "CN khuy nút",
          "CN kiểm hàng", "CN kiểm tra", "Cn Kiểm Tra Chất Lượng", "CN KT Mủ", "CN Kỹ thuật",
          "CN là  gấp  đóng gói (may công nghiệp)", "CN lái máy", "CN Lao động hỗ trợ", "CN Lựa Gỗ",
          "CN lưới không gút", "CN mắc sợi", "CN mài quần bò", "CN may", "CN May", "CN MAY", "cn may",
          "Cn May  KCN Suối Tre  Long Khánh  Đồng Nai", "CN May C3", "CN may CN", "cn may cn", "cn may cn  độc hại",
          "CN May công nghiệp", "Cn May Công Nghiệp", "CN may công nghiệp", "CN Máy cuốn", "CN may trực tiếp CN",
          "CN Mở nắp-Lấy đế - Xưởng Đế PU", "CN mộc máy", "CN nề", "CN PHụ", "CN Phụ May", "CN phụ việc",
          "CN phục vụ sản xuất bia", "CN phun Sơn", "CN qc", "CN rà gạch lò Tuynel TX", "CN ra phôi", "CN Sản xuất",
          "CN sản xuất điện tử", "CN Sản xuất linh kiện điện tử", "CN sản xuất xi măng", "CN SắT", "CN Se sợi",
          "CN SOFA", "CN sơn  in da và pha chế hoá chất để sơn  in da", "CN Sửa chữa",
          "CN sửa chữa cơ điện trong hầm lò", "CN SX các SP làm bằng nhựa Composite", "CN t? in xoa", "CN thành hình",
          "CN Thủ công", "CN Thủ Công", "cn thủ công", "CN thu tiền nước", "CN TMIL B2", "cn toà nhà l",
          "CN Tp HCM - V1 - NV Nhóm Chuyên doanh Sim số", "CN TT trồng rừng và khai thác gỗ", "cn ủi", "CN ủI CN",
          "CN ủi công nghiệp", "CN vận hành băng tải than", "CN vận hành máy may", "cn vận hành máy may",
          "CN vận hành máy may CN", "Cn vận hành máy may công nghiệp", "CN vận hành máy may công nghiệp",
          "cn vận hành máy suốt", "CN vận hành thiết bị", "CN vệ sinh đội VSMT Số 01", "CN VH Dây chuyền sợi",
          "CN Vòi Nước", "CN xưởng gỗ", "CN Xưởng may", "CN. Kiểm Hàng", "CN.Bảo Trì điện",
          "CN.Bảo vệ tuần tra canh gác", "CN_CĂNG", "cnhân", "cnhân đóng nút", "CNKT Điện thoại",
          "CNQLVH Đường Dây & Trạm CNĐ Lục Ngạn", "cnsx", "CNSX  FULLGRAIN", "CNSX chuyền Ferrule", "CNV", "cnv",
          "CN-vệ sinh last", "CNVH/ BQD5", "CNVHMM", "cơ khí sửa chữa", "Công  nhân quét rác đường phố",
          "Công  Nhân Ván", "Công chức", "công chức văn hoá xã hội", "Công Công nhân kiểm tra chất lượng", "Công nhân",
          "công nhân", "Công Nhân", "CÔNG NHÂN", "CôNG NHâN", "công nhân -  xưởng may", "công nhân - Bảo trì (2010)",
          "Công nhân - Bộ phận Sản xuất", "Công nhân - Dán vải", "công nhân  đùn nhựa", "Công nhân  hình ảnh",
          "công nhân  Hoàn Chỉnh", "Công nhân  KTCL Chip 2", "Công nhân - Lắp ráp 2", "Công nhân  may",
          "Công nhân  May", "CÔNG NHÂN - MAY (VITINH-CNGHIEP-CốP Đế)", "Công nhân  May 55",
          "Công Nhân  may công nghiệp", "Công nhân  may công nghiệp", "công nhân  May dép", "Công nhân - Phôi 3",
          "Công nhân  QC", "Công nhân  thủ công", "Công Nhân - Worker", "công nhân (bp eva)",
          "Công nhân (sản xuất nền đĩa thủy tinh) .Phòng Sản xuất", "Công nhân _ Thủ công", "Công Nhân AL2-S1",
          "Công nhân ấn cao tầng", "Công nhân bậc 2", "Công nhân bán lẻ xăng dầu",
          "Công nhân bán lẻ xăng dầu  CHXD Hoàng Đông- Hoàng Đông  Duy Tiên  Hà Nam",
          "Công nhân bán lẻ xăng dầu  CHXD số 5", "Công nhân bán lẻ xăng dầu CHXD số 4", "công nhân băng chuyền",
          "Công nhân bao bản cực", "Công nhân bao bì", "công nhân bao gói 2", "Công nhân bảo trì", "Công Nhân Bavia",
          "Công nhân bế  Bế gia công", "Công nhân bê tông", "Công Nhân Bó Buộc", "Công nhân bỏ dây",
          "Công nhân bộ phận bao gói", "Công nhân bộ phận bao gói mỳ", "Công nhân bộ phận Dập dao thìa dĩa inox",
          "Công nhân bộ phận đế Outsole", "Công nhân bộ phận đế Phylon", "Công Nhân Bộ Phận Dệt",
          "Công nhân bộ phận Đột dập-Dập chuôi", "Công nhân bộ phận đúc", "Công nhân bộ phận Gia công",
          "Công nhân bộ phận kiểm hàng", "Công nhân bộ phận Lắp mới-Máy", "Công nhân bộ phận LEAN 4",
          "CÔNG NHÂN Bộ PHậN NHUộM", "Công nhân bộ phận NOS F", "Công nhân bộ phận phun sơn",
          "Công nhân bộ phận Printer", "Công nhân Bộ PHậN QA", "Công nhân bộ phận QA", "Công nhân Bộ phận Thermal RFID",
          "Công nhân bóc vỏ lụa", "Công nhân Bốc xếp", "Công nhân bốc xếp", "CÔNG NHÂN BỐC XẾP",
          "Công nhân bốc xếp thủ công", "công nhân bốc xếp thủ công", "Công nhân bồn bông", "Công nhân BP Digital",
          "Công nhân Bp In", "Công nhân bp lắp ráp", "Công nhân Cá cắt",
          "Công nhân Cán  luyện cao su trong sản xuất giầy dép", "Công nhân Cán da",
          "Công nhân Cán Tôn-Xưởng Bao Che - Xà Gồ (X3)", "Công nhân cạo mủ", "Công nhân Cạo mủ", "công nhân cạo mủ",
          "Công nhân cạo mủ cao su", "Công Nhân Cao Tốc", "Công nhân cắt", "Công Nhân Cắt", "Công nhân Cắt",
          "Công nhân cát bavia", "Công nhân Cắt Chỉ", "Công nhân cắt gọt vải  ủi  lấy dấu",
          "Công nhân cắt phá  cắt gọt", "Công nhân cắt PVC", "công nhân cắt tỉa", "Công nhân cắt tỉa",
          "Công nhân cắt vải may công nghiệp", "Công nhân cắt vải trong công nghệ may", "Công nhân Chà nhám",
          "Công nhân chà nhám", "Công Nhân Chà Nhám", "Công nhân Chăm sóc", "Công nhân chăm sóc cà phê",
          "công nhân chăn nuôi", "Công nhân chăn nuôi  chăm sóc  vệ sinh chuồng trại gia súc  gia cầm",
          "công nhân chặt", "Công Nhân Chặt", "công nhân chặt tập chung", "Công nhân Chế biến", "công nhân chế biến",
          "Công nhân chế biến hải sản", "Công nhân chế biến mủ cao su", "Công nhân chế biến mủ cao su  XN-CKCB",
          "Công nhân chế biến muối", "Công nhân chế biến thực phẩm", "Công nhân Chế biến thủy  hải sản đông lạnh",
          "Công nhân chế biến thủy hải sản đông lạnh",
          "Công nhân chế biến thủy hải sản đông lạnh- bộ phận dịch vụ cắt tiết", "Công nhân chế biến thủy sản",
          "công nhân chế biến thủy sản", "Công nhân chế biến thủy sản  bậc 2/6-A2-3N2",
          "Công nhân chế biến thủy sản đông lạnh  bậc 1/6", "Công nhân chế tạo", "Công Nhân Chế Tạo",
          "Công nhân chế tạo  gia công  sữa chữa thiết bị sản xuất xi măng - P. cơ khí động lực",
          "Công nhân chi nhánh Phú Quốc", "Công nhân Chia ghép", "Công nhân chiếu xạ", "Công nhân chiếu xạ đế trung",
          "Công nhân Chỉnh Lý", "Công nhân Chỉnh lý-Đế", "Công nhân Chuẩn Bị", "Công nhân chuẩn bị",
          "Công nhân CHUẨN BỊ HÀNG 2", "Công nhân chuyền dập  KCN Long Khánh  thành phố Long Khánh  tỉnh Đồng Nai",
          "Công Nhân chuyền Học Tủ", "Công nhân chuyền K.Harness", "Công Nhân Chuyền Kashito hita",
          "Công nhân chuyền may", "Công Nhân Chuyền May", "Công nhân chuyền men", "Công Nhân Chuyền Thành Hình",
          "công nhân cơ khí", "công nhân Cơ Khí", "Công nhân cơ khí", "Công nhân công nghệ", "công nhân công nghệ",
          "CÔNG NHÂN CÔNG NHÂN QIP", "Công nhân cưa xẻ gỗ", "Công Nhân ĐẠI ĐẾ(ÉP ĐẾ-A)", "Công nhân Dán",
          "Công nhân đan", "Công Nhân Dán Flywire No-Sew", "Công nhân dán lốp", "Công nhân đánh đầu",
          "Công nhân đánh số", "Công nhân Đánh số", "Công nhân đánh số sản phẩm", "Công nhân đào tạo", "Công nhân Dập",
          "CÔNG NHÂN DẬP", "công nhân dập biên", "Công nhân dập móc bin", "Công nhân Dập tay 2/2", "Công nhân Đắp vá",
          "CÔNG NHÂN ĐẾ", "Công nhân Đế", "Công Nhân Dệt", "Công nhân dệt", "Công nhân dệt dây giày",
          "Công nhân dệt may công nghiệp", "Công nhân dệt sợi", "Công Nhân Đi Ca", "Công nhân địa hình",
          "Công nhân điện", "Công nhân Điện", "Công nhân điện bậc 1", "Công nhân điện thêu", "Công nhân điện tử",
          "Công nhân Định hình", "Công nhân định vị", "công nhân ĐM rôbốt", "Công Nhân Đổ PU", "công nhân đổ rót",
          "Công nhân đổi kim", "Công nhân đội vệ sinh", "Công nhân dồn nệm", "CÔNG NHÂN ĐÓNG ĐINH",
          "Công nhân đóng gói", "Công Nhân Đóng Gói", "công nhân đóng gói", "Công nhân Đóng gói",
          "Công nhân đóng gói linh kiện bàn kiếng", "Công nhân đóng kiện", "Công nhân đóng thùng", "Công nhân đốt lò",
          "Công Nhân ĐúC Đế(KP)", "Công nhân Đúc Mài", "Công Nhân Đùn Sợi", "Công Nhân Đùn Sợi", "Công nhân đứng máy",
          "công nhân đứng máy", "Công nhân Đứng Máy", "Công Nhân Đứng Máy", "Công nhân Đứng Máy Chặt",
          "Công nhân đứng máy chạy màng phủ", "Công nhân đứng máy dán - thủ dầu một - bình dương",
          "Công nhân Đứng máy ép đế", "công nhân đứng máy lấy hàng", "Công nhân đứng máy OE",
          "Công nhân đứng máy vạn mã lực", "Công nhân duy tu đường sắt", "Công Nhân ép", "Công nhân ép",
          "Công nhân ép cao tầng", "Công Nhân ép Caochupo", "công nhân ép cọc bê tông cốt thép", "Công nhân ép đế",
          "Công nhân ép đế B", "công nhân ép Đế Đi 3 Ca", "Công nhân ép đế Outsole", "Công nhân ép keo  XN May ĐT",
          "công nhân ép mex (may công nghiệp)", "Công nhân ép MEX bậc 1/6", "CÔNG NHÂN ÉP NHÃN", "Công nhân ép nhựa",
          "Công nhân Fillet", "Công nhân gắn linh kiện", "Công nhân Gắn nhiệt", "Công nhân gấp đóng gói sp",
          "công nhân gấp xếp", "Công nhân gấp xếp trong dây chuyền may công nghiệp", "Công nhân ghép dọc",
          "Công nhân gia công", "Công nhân gia công dây vai", "Công nhân gia công đế",
          "Công nhân gia công linh kiện cao su", "Công nhân Gia Công Ván A", "Công nhân giao nhận", "Công nhân giặt",
          "Công Nhân Giặt Tẩy", "Công nhân giầy", "công nhân giầy", "Công nhân Giầy mẫu", "Công nhân gò",
          "Công nhân gò mũi", "Công nhân hạ cây chặt cành", "Công nhân hàn", "Công nhân Hàn", "Công nhân Hàn cổ góp",
          "Công nhân Hàn điện  hàn hơi", "công nhân hàng trắng cắt", "Công nhân hậu cần sản xuất",
          "Công nhân hình thành", "Công nhân hoàn chỉnh", "Công nhân Hoàn Chỉnh", "Công Nhân Hoàn Chỉnh",
          "Công nhân hoàn chỉnh 3", "Công Nhân Hoàn Chỉnh I", "công nhân hoàn tất", "Công nhân hoàn thành",
          "Công nhân Hoàn Thành", "Công nhân hoàn thiện", "công nhân học việc", "Công nhân in", "Công nhân in caochupo",
          "Công nhân in lụa", "Công nhân In lụa", "Công nhân In Lụa", "Công nhân in lưới gia công",
          "công nhân in offset", "công nhân in sơn", "công nhân KCS", "Công nhân KCS", "Công nhân KCS thành phẩm",
          "Công nhân Khai Phá", "Công Nhân KHAI PHÁT", "công nhân khai thác",
          "Công nhân khai thác khoáng sản trong hầm lò", "Công nhân khai thác mỏ", "Công nhân khai thác mỏ hầm lò",
          "Công nhân khai thác mủ cao su", "công nhân khai thác mủ cao su",
          "Công nhân khai thác mủ cao su - Nông trường cao su Ia Chim", "Công nhân khai thác than hàm lò",
          "Công nhân khai thác than trong hầm lò", "Công nhân khảo sát", "Công nhân khâu may", "Công nhân khâu ráp",
          "công nhân kho", "Công Nhân Kho", "Công nhân kho", "Công nhân Kho Canon", "Công nhân kho hàng",
          "Công nhân kho thành phẩm", "Công nhân kho vật tư", "Công nhân khoan nổ mìn khai thác đá", "Công nhân khuôn",
          "Công nhân Khuôn", "Công nhân khuôn đúc", "Công nhân khuôn đúc - A247237", "Công nhân khuôn đúc - A249231",
          "công nhân khuy nút", "công nhân kiểm hàng", "Công nhân kiểm hàng", "Công Nhân Kiểm hàng",
          "Công Nhân Kiểm Hàng", "Công nhân Kiểm hàng", "công nhân kiểm hàng khâu may", "Công nhân kiểm hóa",
          "Công nhân Kiểm hóa  ủi  đóng gói gấp cắt chỉ  so sữa BTP", "Công nhân kiểm phẩm", "Công nhân kiểm tra",
          "công nhân kiểm tra", "Công Nhân Kiểm Tra Chất Lượng", "Công nhân kiểm tra chất lượng",
          "Công nhân Kiểm tra chất lượng", "Cộng Nhân Kiểm tra chất lượng - IP",
          "Công nhân kiểm tra chất lượng sản phẩm", "Công nhân kiểm tra chất lượng van", "công nhân kiểm tra sản phẩm.",
          "Công nhân kinh doanh  Điện lực Can Lộc", "Công nhân kinh doanh - Điện lực TP Vinh", "công nhân ktclsp",
          "Công nhân kỹ thuật", "công nhân kỹ thuật", "Công nhân- kỹ thuật", "Công nhân kỷ thuật",
          "Công Nhân Kỹ Thuật Đánh Bóng", "Công nhân kỹ thuật điện", "Công nhân kỹ thuật đóng gói",
          "Công nhân kỹ thuật may", "công nhân kỹ thuật. 1.14  bậc 1/15", "Công nhân là", "Công nhân là",
          "Công nhân Là", "Công nhân là  bộ phận chuyền may", "Công nhân là  gấp  đóng gói",
          "Công nhân là  gấp đóng gói sản phẩm may", "Công nhân là (May công nghiệp)", "Công nhân là bậc 1/6",
          "Công nhân là hơi", "Công Nhân Là Hơi", "Công nhân là vải may công nghiệp",
          "Công nhân lái máy nông nghiệp-Xí nghiệp Cơ giới Nông nghiệp", "Công nhân lái xe", "Công nhân Lái xe",
          "công nhân Lái xe", "Công nhân lái xe cẩu", "Công nhân lái xe tải đội xe từ 7.5 tấn đến 16.5 tấn",
          "Công nhân lái xe tải P>= 7.5 tấn", "Công nhân làm đường", "Công nhân làm lưới vườn",
          "Công Nhân Làm Sạch Bằng Mek", "Công nhân làm việc theo chế độ 03 ca", "Công nhân Lăn trở bánh phồng tôm",
          "Công nhân lành nghề", "Công nhân lao động phổ thông", "Công nhân lắp dựng", "Công nhân lắp ráp",
          "công nhân lắp ráp", "CÔNG NHÂN LẮP RÁP", "Công Nhân Lắp Ráp",
          "Công nhân lắp ráp  vận chuyển linh kiện  kiểm tra và đóng gói sản phẩm.", "Công Nhân Lắp Ráp 420",
          "Công Nhân Lắp Ráp Bình Acquy", "Công nhân lắp ráp điện tử", "Công Nhân lắp ráp điện tử",
          "CÔNG NHÂN LẮP RẮP ĐIỆN TỬ", "CÔNG NHÂN LẮP RÁP ĐIỆN TỬ;", "Công nhân lắp ráp linh kiện điện tử",
          "Công nhân lắp ráp mạch điện tử", "công nhân LắP RáP MOTOR", "công nhân lắp ráp nhà máy động cơ",
          "Công nhân Lắp ráp sản phẩm", "Công nhân LĐPT quét rác đường phố- bậc 1/7 - hệ số 1 31", "Công nhân LDS",
          "Công nhân lò hấp cá", "Công nhân lò nung", "Công nhân Lò sấy", "Công nhân lộn găng", "Công nhân mạ đồng",
          "công nhân mài", "công nhân mài bóng", "Công nhân Mài bụi đế lớn", "Công nhân mài đế lớn",
          "Công nhân mài eva", "Công nhân mài ngồi", "Công Nhân May", "Công nhân may", "công nhân may", "Công nhân May",
          "Công Nhân may", "CÔNG NHÂN MAY", "công nhân May", "Công Nhân May 1 Kim", "Công nhân may bậc 1/12",
          "Công nhân máy cán", "Công nhân máy cắt", "Công nhân May CN", "Công nhân may cn", "Công nhân may CN",
          "công nhân may cn", "công nhân may công nghiệp", "Công Nhân May Công Nghiệp", "Công nhân may công nghiệp",
          "Công Nhân may công nghiệp", "Công nhân May công nghiệp", "CÔNG NHÂN MAY CÔNG NGHIệP",
          "Công nhân may Công Nghiệp", "Công Nhân May công nghiệp", "Công nhân may công nghiệp - LINE 12",
          "Công nhân may công nghiệp- Nhà máy May 1", "Công nhân máy ép", "CÔNG NHÂN MÁY ÉP", "Công nhân may giày",
          "Công Nhân May Khâu 4 --AD2FP02D", "Công nhân may Mẫu  XN may ĐT", "Công nhân may máy",
          "Công nhân may máy may công nghiệp", "Công nhân May máyA05", "Công nhân Máy nhựa", "CÔNG NHÂN MAY NóN",
          "công nhân may phân xưởng may II", "Công nhân may tay", "Công Nhân May Thêu", "Công nhân máy thêu khác",
          "Công Nhân May Trụ 1 Kim", "Công Nhân May Vi Tính", "Công nhân maycông nghiệp", "CÔNG NHÂN MDGC",
          "Công nhân Mihonchou", "Công nhân mộc", "Công nhân MỘC 1", "công nhân mực", "Công Nhân Nấu ăn",
          "công nhân nấu liệu", "công nhân nề hoàn thiện", "Công nhân nghiền bột giấy bậc 4/7", "Công nhân nghiền đá",
          "Công nhân nghiền phụ gia  vê viên  sấy  đóng bốc bao NPK", "Công nhân ngồi máy lạng da",
          "Công nhân nguội sửa chữa toa xe lửa", "Công nhân nhà máy (SX găng tay da)", "Công nhân nhà máy Bao bì giấy",
          "CÔNG NHÂN NHUỘM", "Công nhân Nhuộm", "Công nhân nhuộm", "Công nhân Nobashi", "Công nhân NƯớC RửA 2",
          "công nhân ống", "Công nhân oze", "Công Nhân P.thí Nghiệm", "công nhân pha cắt", "Công nhân pha keo",
          "Công nhân Pha keo", "Công Nhân Pha Keo", "Công nhân pha màu", "Công nhân pha phao",
          "Công nhân pha phao nhỏ A", "Công nhân pha phao nhỏ B", "Công nhân pha phao nhỏ C",
          "Công nhân pha trộn cát làm khuôn đúc", "Công nhân phân loại thép phế liệu bậc 1/7  phân xưởng nguyên liệu",
          "Công nhân phân xưởng Dỡ - Phân loại sản phẩm", "Công nhân phân xưởng gia công",
          "Công Nhân Phân Xưởng Hoàn Chỉnh", "công nhân phân xưởng kiểm nghiệm", "Công nhân phân xưởng ống đen",
          "Công nhân phổ thông", "Công nhân phơi đão", "Công nhân phối giày", "Công Nhân Phối Hàng",
          "CÔNG NHÂN PHốI TRộN", "Công nhân phòng chế tạo", "Công nhân Phòng chuẩn bị 2",
          "Công nhân Phòng Đảm bảo chất lượng Tủ lạnh", "Công Nhân Phòng In Nhựa", "Công nhân phòng khuôn",
          "công nhân phòng mẫu", "Công Nhân Phòng Mẫu (QC)", "Công nhân phòng máy", "Công nhân PHòNG MAY MáY 1- AW",
          "Công Nhân Phòng Pha Màu", "Công nhân phòng QC", "Công nhân phòng QUảN Lý SảN XUấT", "Công nhân phòng Sạch",
          "Công nhân phòng Sản xuất", "Công nhân phòng sản xuất Ato", "Công nhân phụ", "Công nhân phụ  cắt giầy nữ",
          "Công nhân phụ chuyền may", "Công Nhân Phụ Hoàn Thành", "công nhân phụ kho", "Công nhân phụ may",
          "Công nhân Phụ May -4005", "Công nhân phụ máy bồi dán", "công nhân phụ may công nghiệp",
          "Công nhân phụ may công nghiệp", "CÔNG NHÂN PHụ MAY CÔNG NGHIệP", "Công nhân phụ tổ inox",
          "Công nhân phụ việc", "Công nhân Phục vụ", "công nhân phục vụ phụ trợ", "Công nhân Phun sơn",
          "Công nhân QA - OQC - R2", "Công nhân QC", "Công nhân QC dệt", "Công Nhân Qc1", "công nhân QC2",
          "công nhân quản lý đường dây và trạm chi nhánh điện tiên phước",
          "Công nhân quản lý và khai thác đèn biển trên các đảo và cửa biển",
          "Công nhân quản lý vận hành lưới điện trung  hạ thế - Điện lực Sông Mã", "công nhan quét dọn",
          "Công Nhân Quét Keo", "Công nhân quét keo dán chi tiết", "Công nhân Ra gỗ-B", "Công nhân Ráp 1",
          "Công Nhân Roll - Outside", "Công nhân sản xuât", "Công nhân sản xuất", "công nhân sản xuất",
          "Công nhân Sản Xuất", "Công nhân Sản xuất", "CÔNG NHÂN SẢN XUẤT", "Công Nhân Sản xuất",
          "Công Nhân sản xuất", "Công Nhân Sản Xuất", "Công nhân sản xuất  Bộ phận sơn gia công",
          "CÔNG NHÂN SẢN XUẤT BAO BÌ", "công nhân sản xuất bê tông", "Công nhân sản xuất bộ phận sơn",
          "Công nhân sản xuất Camera", "Công nhân sản xuất camera điện thoại", "công nhân sản xuất cần câu cá",
          "Công nhân sản xuất dây dẫn điện", "Công nhân sản xuất dây điện", "Công Nhân Sản Xuất dây kéo",
          "Công Nhân sản xuất Điện tử", "CÔNG NHÂN SảN XUấT ĐIệN Tử", "Công nhân sản xuất điện tử",
          "Công Nhân Sản Xuất Điện tử", "Công nhân sản xuất đồ dùng bằng da bậc 1/6 N2A2.2",
          "Công Nhân Sản Xuất Gạch Ngói Tổ ép Máy", "Công nhân Sản xuất gạch ốp lát", "Công nhân sản xuất giầy",
          "Công Nhân Sản Xuất gỗ", "Công Nhân Sản Xuất Hạt Nhựa", "Công nhân sản xuất khăn ướt",
          "công nhân sản xuất lắc", "Công nhân sản xuất lắp ráp", "Công nhân Sản xuất lắp ráp",
          "Công nhân Sản xuất lắp ráp Máy giặt", "Công nhân Sản xuất lắp ráp Tủ lạnh",
          "Công nhân sản xuất linh kện điện tử", "Công nhân sản xuất linh kiện Camera",
          "Công nhân sản xuất linh kiện điện thoại", "Công nhân sản xuất linh kiện điện tử",
          "Công nhân Sản xuất linh kiện Điện tử", "Công nhân sản xuất máy giặt", "Công nhân sản xuất nền đĩa thủy tinh",
          "Công nhân sản xuất nhựa", "Công nhân sản xuất phân bón", "Công Nhân Sản Xuất Phốt",
          "Công nhân sản xuất phụ kiện điện thoại", "Công nhân sản xuất roong", "Công nhân sản xuất sản phẩm sơn mạ",
          "công nhân sản xuất sợi", "Công nhân sản xuất sơn", "Công nhân sản xuất tại KCN NT1",
          "Công nhân sản xuất thiết bị điện tử", "Công nhân sản xuất Tôm", "Công nhân sản xuất túi khí vô lăng ô tô",
          "Công nhân sản xuất xà phòng bột  xà phòng kem", "Công nhân sản xuất xưởng tiện", "Công nhân Sản xuất/PCB/MC",
          "Công nhân sản xuất/Worker", "Công nhân sắp việc", "Công nhân Sắp xếp hàng lên container",
          "Công nhân- Sắp xếp khuôn", "Công nhân sấy", "Công nhân Sấy lò", "Công nhân sinh quản",
          "Công nhân sơ chế dừa", "Công nhân sofa", "công nhân sợi", "công nhân SƠN", "Công nhân Sơn", "Công nhân sơn",
          "công nhân sơn", "CÔNG NHÂN SƠN", "Công nhân Sơn B", "Công nhân sơn bằng phương pháp thủ công",
          "CÔNG NHÂN SƠN BóNG", "Công nhân sơn in da và pha chế hóa chất để sơn  in da", "Công Nhân Sơn Mài",
          "Công nhân sơn tay", "Công nhân Sửa cá", "Công Nhân sửa cá", "Công nhân sửa cá", "Công nhân sữa chữa",
          "Công nhân sửa chữa", "công nhân sửa chữa cơ điện trong hầm lò", "Công nhân sửa chữa cơ khí mỏ",
          "Công nhân sửa chữa đường sắt", "Công nhân sửa chữa máy may",
          "Công nhân sửa chữa thiết bị công trường khai thác bậc 6/7", "Công nhân sushi", "Công nhân Sushi",
          "Công nhân SX", "công nhân sx", "Công Nhân SX Điện tử", "Công nhân SX hình ảnh", "Công Nhân SX thiết bị y tế",
          "Công nhân T10MAY3", "Công nhân T12MAY3", "Công nhân T13MAY3", "Công nhân tạo hình", "Công nhân Tẩy hàng",
          "Công nhân Thành hình", "công nhân thành hình", "Công nhân thành hình", "Công nhân Thành Hình",
          "Công Nhân Thành Hình", "Công nhân thành hình  KCN Long Khánh  thị xã Long Khánh  tỉnh Đồng Nai",
          "Công nhân Tháo phom", "Công Nhân Thao Tác", "Công nhân thao tác", "Công nhân thao tác công đoạn chuẩn bị",
          "Công nhân thao tác máy kéo", "Công nhân Thao tác máy phun đúc", "Công nhân thay thân", "Công nhân thêu",
          "Công nhân thi công", "Công Nhân Thợ May", "Công nhân thợ may yếu", "Công nhân thợ phụ-Tổ 02",
          "Công nhân thợ sơn", "Công nhân thổi in PE", "Công nhân thời trang", "Công nhân Thời trang",
          "Công nhân Thủ công", "Công nhân thủ công", "Công nhân Thủ Công", "Công nhân thủ công (Kiểm hàng)",
          "công nhân thủ công gia công lần 2", "Công nhân thủ công may", "Công nhân thủ công-Phân xưởng Long Xuyên",
          "Công nhân thủ kho", "Công nhân thực thi cung ứng", "công nhân tinh chế sản phẩm", "Công nhân tổ BTP",
          "Công nhân tổ cắt", "Công nhân tổ Cắt da", "Công nhân tổ chuyền ngoài", "Công nhân tổ cơ khí",
          "Công nhân Tổ công vụ", "Công Nhân Tổ ĐàO TạO", "Công nhân tổ dệt", "Công nhân tổ ép phun 2",
          "Công nhân Tổ Lắp Ráp 4", "công nhân tổ lò", "Công nhân tổ may", "Công nhân tổ May mũ giày",
          "Công nhân tổ nung đốt", "Công nhân tổ Quấn chão", "Công nhân tổ Ráp giày thành phẩm",
          "Công nhân tổ thành hình", "Công nhân Tổ Thành phẩm", "Công nhân Tổ Tinh 5", "công nhân toà nhà l",
          "Công nhân Trại Gà", "Công nhân trại heo", "Công Nhân Trại Heo", "Công nhân trãi vải", "Công nhân trải vải",
          "Công nhân trải vải  vận hành máy cắt vải", "Công Nhân Trải Vải(MCN)",
          "Công nhân trang trí bề mặt gỗ  bậc 1/7", "Công nhân trộn liệu", "Công nhân trồng  chăm sóc cao su",
          "Công nhân trồng chè", "Công nhân trồng lúa", "công nhân trồng mía", "Công nhân trồng rừng",
          "Công nhân trồng và thu hoạch", "Công nhân trực tiếp sản xuất",
          "Công nhân trực tiếp sản xuất  làm việc theo chế độ 3 ca", "Công Nhân ủi", "Công nhân ủi",
          "Công Nhân ủi Công Nghiệp", "Công nhân ủi may công nghiệp", "Công nhân Upper", "Công nhân vá lưới",
          "Công nhân vận chuyển vải may CN", "công nhân ván ép", "Công nhân vận hành",
          "Công nhân vận hành - Thường xuyên tiếp xúc hóa chất độc hại - Công ty TNHH Aica Đồng Nai",
          "Công nhân vận hành bậc 3/7 - Trạm thủy nông Thanh Sơn", "Công nhân vận hành bơm nước",
          "Công nhân vận hành dây chuyền kéo sợi", "Công nhân vận hành dây chuyền sợi", "Công nhân vận hành lò hơi",
          "công nhân vận hành lò hơi tại thanh xuân   Hà nội", "Công nhân vận hành máy bơm công suất>8000m3",
          "Công nhân vận hành máy cắt", "Công nhân Vận Hành máy Công ty Thanh Sơn Hóa Nông",
          "Công nhân vận hành máy dệt khí", "Công nhân vận hành máy gia công kim loại",
          "Công nhân Vận hành máy kết đáy", "Công nhân vận hành máy lạnh",
          "Công nhân vận hành máy mắc  máy hồ vải  sợi trong dây chuyền dệt", "Công nhân vận hành máy may",
          "Công nhân Vận hành máy may", "Công Nhân Vận Hành Máy May", "CÔNG NHÂN VẬN HÀNH MÁY MAY",
          "Công nhân vận hành máy may - Line 38", "Công nhân vận hành máy may (may công nghiệp)",
          "Công nhân vận hành máy may CN", "Công nhân vận hành may may công nghiệp",
          "Công nhân vận hành máy may công nghiệp", "Công nhân vận Hành máy may công nghiệp",
          "Công Nhân vận hành máy May Công nghiệp", "Công Nhân Vận Hành Máy may công nghiệp",
          "công nhân vận hành máy may công nghiệp", "Công nhân vận hành máy may( May công nghiệp)",
          "Công Nhân Vận Hảnh Máy May|(MCN)", "Công nhân vận hành máy nghiền tuyển quặng sắt   CN",
          "Công nhân vận hành máy sản xuất van", "công nhân vận hành máy thêu",
          "Công nhân vận hành Nhà máy Thủy điện Ea Krông Run", "Công nhân Vận hành Thành Vinh Ca A",
          "Công nhân vận hành thiết bị", "Công Nhân Vận hành thiết bị máy",
          "Công nhân vận hành thiết bị may công nghiệp", "Công nhân vận hành thiết bị may công nghiệp  bậc 1/6",
          "Công nhân vận hành thiết bị xe nâng", "Công nhân vận hành XN sản xuất KD nước sạch Bảo Thắng",
          "Công nhân vẽ", "Công Nhân Vệ sinh", "Công nhân vệ sinh", "Công nhân vệ sinh giày",
          "Công nhân vệ sinh môi trường", "Công nhân VH thiết bị", "Công nhân VHBĐ", "công nhân VHTB lưu hóa lốp",
          "công nhân viên", "Công nhân viên", "Công Nhân Viên", "Công nhân vô gòn 2", "công nhân vô lon cá",
          "Công nhân VSMT", "công nhân xếp bản cực", "Công nhân xếp bao", "Công nhân Xếp gỗ Tổ Kho gỗ",
          "Công nhân Xếp hộp", "Công nhân Xếp nguyên vật liệu", "Công nhân Xi mạ", "công nhân xi mạ",
          "Công nhân xí nghiệp CTN - CTĐT số 5", "công nhân xỏ dây", "công nhân xưởng 7 ép nhôm", "Công nhân xưởng cá",
          "Công nhân xưởng cắt", "Công nhân xưởng cường hóa", "Công nhân Xưởng gia công", "Công nhân Xưởng In",
          "Công nhân xưởng may", "Công Nhân Xưởng may", "Công nhân xưởng Tráng vải bạt", "công nhân/ép 2",
          "Công nhân: Hàn điện  hàn hơi", "Công Nhân-Chuyền Thành Hình", "Công nhân-dán da cố", "Công nhân-Dán gia cố",
          "Công nhân-điều phối hàng-lãnh hàng", "Công nhân-Đóng gói", "Công nhân-đứng máy IP",
          "Công nhân-ép tem đệm lót", "Công nhân-GEA922", "công nhân-HĐ 01 năm", "Công nhân-in hóa chất",
          "công nhân-may công nghiệp", "Công nhân-may công nghiệp", "Công NhânOrA3-Job", "Công nhân-Thủ công",
          "CÔNG NHÂN-Tổ cào sớ - Xưởng 3", "Cơng nhn bộ phận May Long Thnh", "Coõng nhaõn boọ phaọn Nos E",
          "Coõng nhaõn boọ phaọn ủeỏ Phylon", "Coõng nhaõn Eựp suaỏt thaứnh hỡnh caực saỷn phaồm cao su",
          "Coõng nhaõn Pha cheỏ  san roựt hoựa chaỏt", "Coõng nhaõn trửùc tieỏp maựy may coõng nghieọp",
          "Cử nhn điều dưỡng", "C.n", "C.n Cty Tnhh May Mặc Wonderful Sài Gòn", "C.n May", "C.nhân  may",
          "C.Nhân chất lượng", "C.nhân kiểm hàng", "C.nhân may", "C.nhân phụ may", "C.Nhân Sx", "C.nhân trãi vải"]

    v3 = ["Cán bộ", "cán bộ", "Cán bộ  Văn phòng thống kê", "Cán bộ ban tuyên giáo", "Cán bộ cấp 2",
          "cán bộ công nhân viên", "Cán bộ dân số - Kế hoạch hoá gia đình Trạm y tế xã An Thành",
          "Cán bộ Dân Số KHHGĐ-", "Cán bộ KD kiểm tra KHSDĐ", "Cán bộ không chuyên trách", "Cán bộ kỹ thuật",
          "Cán bộ nhân viên", "Cán bộ Tài chính - Kế toán", "Cán bộ thanh tra", "Cán bộ tín dụng", "Cán bộ tư pháp",
          "Cán bộ văn hóa xã hội", "Cán bộ văn hoá xã hội.", "cán bộ văn phòng", "Cán bộ văn phòng",
          "cán bộ văn phòng - thống kê", "cán bộ văn phòng thống kê", "Cán bộ văn phòng thống kê", "Cán bộ VP - TK"]

    v4 = ["Chủ Tịch", "chủ tịch hội CCB", "Chủ tịch hội LHPN xã", "Chủ tịch Hội liên hiệp phụ nữ",
          "Chủ tịch Hội phụ nữ", "Chủ tịch hội phụ nữ", "Chủ tịch UBMT TQVN xã"]

    v5 = ["Chuyên viên", "chuyên viên - BHXH Mai Sơn", "Chuyên viên  Khách hàng doanh nghiệp vi mô",
          "Chuyên viên  Phòng Cấp sổ thẻ", "Chuyên viên - Phòng Quản lý Thu", "Chuyên Viên Bậc 1/8",
          "Chuyên viên Địa chính tài nguyên môi trường", "Chuyên viên Địa chính- xây dựng", "Chuyên viên Dự toán",
          "Chuyên viên hành chính", "Chuyên viên kỹ thuật dự án modul hóa", "Chuyên viên nghiệp vụ depot",
          "Chuyên viên phiên dịch tiếng Nhật.", "Chuyên viên Phòng Bưu chính viễn thông", "Chuyên viên phòng KHKT-VT",
          "Chuyên viên phục vụ tại chỗ", "Chuyên viên tuyển sinh", "Chuyên viên xét khiếu tố",
          "chuyên viên-BHXH huyện Thạnh Hoá"]

    v7 = ["Điều dưỡng", "Điều Dưỡng", "Điều dưỡng  chuyên cấp cứu -  Khoa Hồi sức cấp cứu",
          "Điều dưỡng hạng IV  bậc 1/12", "Điều dưỡng hạng IV  Khoa Chẩn đoán hình ảnh  Bệnh viện ĐKKV Bắc Quảng Bình",
          "Điều dưỡng khoa Nội - Nhi - Nhiễm", "Điều dưỡng TC", "Điều dưỡng trung cấp", "Điều Dưỡng Trung Cấp",
          "Điều dưỡng trung cấp Nhân viên y tế học đường", "Điều dưỡng trung học", "Điều dưỡng trung học  khoa khám",
          "Điều dưỡng trung học  khoa Ngoại tổng hợp",
          "Điều dưỡng Trung học (Trực tiếp khám  điều trị  phục vụ bệnh nhân ở các Khoa/Phòng khám bệnh  cấp cứu tổng hợp của bệnh viện) - Khoa Khám bệnh",
          "Điều dưỡng trung học Khoa Phẫu Thuật Gây Mê Hồi Sức", "Điều dưỡng trung học Lão khoa",
          "Điều dưỡng trung học trạm y tế Xã Hội Sơn", "Điều dưỡng trưởng",
          "Điều dưỡng trưởng  Khoa Kiểm soát nhiễm khuẩn", "Điều dưỡng viên", "điều dưỡng viên"]

    v8 = ["DƯợC Sĩ", "Dược sĩ", "dược sĩ", "Dược sĩ trung học", "Dược sỹ", "Dược Sỹ Bán Hàng", "Dược sỹ trung cấp",
          "dược sỹ trung học", "Dược sỹ trung học cơ sở cấp phát thuốc Methadone Xã Ma Thì Hồ"]

    v9 = ["Giám đốc", "Giám Đốc", "giám đốc", "Giám đốc chi nhánh",
          "Giám đốc phòng giao dịch Agribank chi nhánh huyện Hưng Nguyên",
          "Giám đốc Quan hệ khách hàng - RB (phi tín dụng) TP CAO LÃNH - ĐỒNG THÁP", "hiệu trưởng", "Hiệu trưởng"]

    v10 = ["Giám sát", "Giám sát bán hàng", "Giám sát bán hàng - Kênh TT- L.1", "Giám sát buồng phòng",
           "Giám Sát Công Trình", "Giám sát kinh doanh", "Giám sát KV", "Giám sát kỹ thuật", "Giám sát nội thất",
           "Giám sát thi công", "Giám sát viên bảo vệ", "Giám sát viên Tập sự"]

    v11 = ["Giáo viên", "giáo viên", "Giáo  viên", "Giảng viên","Giáo Viên", "GIÁO VIÊN", "Giáo viên  kiêm tổ trưởng",
           "Giáo viên - TH số 1 Nậm Xây", "GIáo viên - tổ phó", "Giáo viên  tổ trưởng",
           "Giáo viên - Trường Mẫu Giáo Vĩnh Viễn 2", "Giáo viên (Tổ phó)",
           "Giáo viên dạy TH  Lái xe  TT ĐT Lái xe ô tô Cơ giới", "giáo viên dạy thực hành", "Giáo viên kiêm tổ trưởng",
           "giáo viên mầm non", "Giáo viên mầm non", "Giáo viên Mầm non", "Giáo viên mầm non (Lao động hợp đồng)",
           "Giáo viên Mầm non bậc  2/12 .Trường MN Bế Triều", "Giáo viên Mầm non Bình Lư",
           "Giáo viên MN - Tr  MN Thị Trấn", "Giáo viên MN. Tr MN Hoa Ban Mường Bang", "Giáo viên THCS",
           "Giáo viên THCS bậc 1", "giáo viên THCS chính", "Giáo viên THCS Chính", "Giáo viên THCS chính",
           "Giáo viên THCS Liên Hoà", "giáo viên THCS quyết tiến", "Giáo viên Thể dục", "Giáo viên THPT",
           "Giáo viên tiểu học", "giáo viên tiểu học cao cấp bậc 1", "Giáo viên tiểu học chính",
           "Giáo viên tiểu học chính bậc 1/10  Trường PTCS Vĩnh Phong thuộc", "Giáo viên Tiểu học hạng  II",
           "Giáo viên trợ giảng", "giáo viên trung học", "Giáo viên trung học", "Giáo viên trường  mẫu giáo Tân Bình 2",
           "Giáo viên trường mầm non xã Liên Bảo", "Giáo viên trường MG Tân Bình 2", "Giáo viên Trường TH Hồ Tùng Mậu",
           "giáo viên trường THCS nguyễn huệ", "Giáo viên trường THCS Thạnh Hoà", "Giáo viên trường tiểu học",
           "Gíao viên trường tiểu học  tân phước 1", "Giáo viên trường tiểu học Kim Đồng  TT Chư Ty", "Giáo viên.",
           "Giáo viên-Tổ trưởng", "GV", "GV Trường MG Nậm Mòn", "GV-TH LKA1"]

    v12 = ["Nhan Vien", "nhan vien", "Nhân Viên","Nhân  viên", "Nhân viêm", "Nhan vien", "Nhân Viên", "Nhân viên",
           "Nhân viên - Hà Nội", "Nhân Viên  kỹ sư xây dựng", "Nhân viên  Kỹ thuật", "Nhân viên - Lái xe xe nâng",
           "Nhân viên - V01 - Tỉnh Bình Dương", "Nhân viên - V01 - Tỉnh Đồng Nai", "Nhân viên - V01 - Tỉnh Hải Phòng",
           "Nhân viên - V01 - TP. Hà Nội", "Nhân viên - V01 - TP. Hải Phòng", "Nhân viên - V01 - TP. Hồ Chí Minh",
           "Nhân viên - V02 - CN Cần Thơ", "Nhân viên - V02 - Tỉnh Bình Phước", "Nhân viên - V02 - Tỉnh Cà Mau",
           "Nhân viên - V02 - Tỉnh Cần Thơ", "Nhân viên - V02 - Tỉnh Hải Dương", "Nhân viên - V02 - Tỉnh Kiên Giang",
           "Nhân viên - V02 - Tỉnh Long An", "Nhân viên - V02 - Tỉnh Quảng Ninh", "nhân viên", "NHÂN VIÊN",
           "Nhân viên - V02 - Tỉnh Thừa Thiên Huế", "Nhân viên - V02 - Tỉnh Tiền Giang", "Nhân viên - V03 - Nghệ An",
           "Nhân viên - V03 - Quảng Nam", "Nhân viên - V03 - Tỉnh An Giang", "Nhân viên - V03 - Tỉnh Bắc Giang",
           "Nhân viên - V03 - Tỉnh Cao Bằng", "Nhân viên - V03 - Tỉnh Đồng Tháp", "Nhân viên - V03 - Tỉnh Nam Định",
           "Nhân viên - V03 - Tỉnh Ninh Thuận", "Nhân viên - V03 - Tỉnh Quảng Nam", "Nhân viên - V03 - Tỉnh Thanh Hóa",
           "Nhân viên - V03 - Tỉnh Yên Bái", "Nhân viên - V04 - CN Nghệ An", "Nhân viên - V04 - Tỉnh Bình Định",
           "Nhân viên - V04 - Tỉnh Bình Thuận", "Nhân viên - V04 - Tỉnh Đắk Nông", "Nhân viên - V04 - Tỉnh Lai Châu",
           "Nhân viên - V04 - Tỉnh Lạng Sơn", "Nhân viên - V04 - Tỉnh Nghệ An", "Nhân viên - V04 - Tỉnh Ninh Thuận",
           "Nhân viên - V04 - Tỉnh Sơn La", "Nhân viên - V04 - Tỉnh Thanh Hóa", "Nhân viên - V04 - Tỉnh Vĩnh Long",
           "Nhân viên An Ninh", "Nhân viên An toàn lao động", "Nhân viên bậc 2/12", "Nhân viên bán hàng",
           "Nhân Viên Bán Hàng", "Nhân viên Bán hàng", "Nhân Viên bán Hàng", "nhân viên bán hàng", "Nhân Viên Bán hàng",
           "nhân viên Bán hàng", "Nhân Viên Bán Hàng - Bình Định - Kv4", "Nhân Viên Bán Hàng - Điện Biên - Kv4",
           "nhân viên bán hàng - kênh tt - l.1", "Nhân Viên Bán Hàng - Kiên Giang - Kv2",
           "Nhân Viên Bán Hàng - Kon Tum - Kv3", "Nhân Viên Bán Hàng - Quảng Ngãi - Thành Phố Quảng Ngãi - Kv3",
           "nhân viên bán hàng - Takashimaya", "Nhân Viên Bán Hàng - Thanh Hoá - Thành Phố Thanh Hóa - Kv3",
           "Nhân Viên Bán Hàng - Tp. Hà Nội - Kv1", "Nhân Viên Bán Hàng - Tp. Hồ Chí Minh - Kv1",
           "Nhân Viên Bán Hàng - Yên Bái Kv4", "Nhân viên bán hàng cao cấp", "Nhân viên bán hàng Chi nhánh Tiền Giang",
           "Nhân viên bán hàng NGHE AN/vùng 2", "Nhân viên bán hàng NPP", "Nhân Viên Bán Hàng Qua Điện Thoại",
           "Nhân viên bán hàng Quầy Viễn thông 35", "Nhân viên bán hàng tại cửa hàng xăng dầu",
           "Nhân viên bán hàng tại Lào Cai- Vùng 2", "Nhân viên bán hàng trực tiếp", "Nhân Viên Bán Hàng Trực Tiếp",
           "Nhân viên Bán hàng VinMart+", "Nhân Viên Bán Hàng-Hcmc 1", "Nhân viên bán vé",
           "Nhân viên bán vé trên xe buýt", "Nhân viên bán vé xe buÝt", "nhân viên bán xăng",
           "Nhân Viên Báo Cáo Số Liệu", "Nhân viên Bào chế", "Nhân viên Bảo hành", "nhân viên bảo mẫu",
           "nhân viên bảo toàn sửa chữa máy may", "Nhân viên bảo trì", "Nhân viên Bảo Trì", "Nhân Viên Bảo Trì",
           "nhân viên bảo trì máy may", "nhân viên bảo vệ", "Nhân viên bảo vệ", "Nhân viên Bảo vệ",
           "Nhân viên Bảo vệ   Lái xe", "Nhân viên Bảo vệ - B4.N2 1/5", "Nhân viên Bảo vệ chuỗi", "Nhân viên Bếp",
           "Nhân viên bếp", "nhân viên bếp", "Nhân viên Bếp chính", "Nhân viên Bếp Hoa", "Nhân viên bộ phận khuôn",
           "Nhân viện bộ phận Mold công đoạn 7-1 7-2", "Nhân Viên Bộ Phận Nos J", "Nhân viên Bốc dỡ",
           "Nhân viên Bốc Xếp", "Nhân viên bốc xếp thủ công ở các kho", "nhân viên BP kiểm phẩm", "Nhân viên Buồng",
           "Nhân viên buồng", "Nhân viên buồng phòng", "Nhân Viên Buồng Phòng", "Nhân viên Cắt may",
           "Nhân viên Chăm Sóc Khách Hàng", "Nhân viên chăm sóc sắc đẹp", "Nhân viên chăm sóc trại heo",
           "Nhân viên chế biến thực phẩm", "Nhân viên Chi nhánh Văn phòng đăng ký đất đai huyện Mỏ Cày Bắc",
           "Nhân viên chi trả-01", "Nhân viên CND Thanh Sơn", "Nhân viên Cơ điện", "Nhân viên cơ điện",
           "Nhân viên cơ khí chỉnh máy", "Nhân viên Công nghệ",
           "NHÂN VIÊN CÔNG TY TNHH TRUYỀN HÌNH CÁP SAIGONTOURIST. ĐỊA CHỈ: 31-33 ĐINH CÔNG TRÁNG PHƯỜNG TÂN ĐỊNH  QUẬN 1  TPHCM",
           "Nhân viên công vụ", "Nhân viên CR", "Nhân viên CSKH",
           "Nhân viên cty TNHH MTV TM AI SIN - Nhân viên bán hàng", "Nhân viên cung ứng",
           "Nhân viên đào tạo kĩ thuật (Phòng Đóng Gói)", "Nhân viên dịch vụ khách hàng",
           "Nhân viên Dịch vụ khách hàng - Thu ngân", "Nhân viên Dịch vụ sau bán hàng - V01 - Q.Thủ Đức - Hồ Chí Minh",
           "Nhân viên Dịch vụ sau bán hàng - V03 - Vị Thanh - Hậu Giang", "nhân viên dịch vụ thẩm mỹ",
           "Nhân viên điểm giới thiệu dịch vụ (Điện máy)",
           "Nhân viên điểm giới thiệu dịch vụ (Điện máy) - Nơi làm việc: Bình Dương",
           "Nhân viên điểm giới thiệu dịch vụ (Điện máy) - Nơi làm việc: Hồ Chí Minh",
           "Nhân viên điểm giới thiệu dịch vụ (Xe máy)",
           "Nhân viên điểm giới thiệu dịch vụ (Xe máy) - Nơi làm việc: Bắc Giang", "nhân viên điều hành",
           "Nhân viên Điều Hành Điểm Tiếp Thị", "Nhân viên điều hành tuor kiêm thủ quỹ",
           "Nhân Viên Điều Tra - Miền Nam", "Nhân viên đổ sợi sợi con", "Nhân viên đội kiểm lâm cơ động",
           "Nhân viên đóng bao thành phẩm", "Nhân viên đóng gói", "Nhân viên đứng máy", "Nhân viên đứng máy chải kỹ",
           "Nhân viên đứng máy ghép", "Nhân viên đứng máy OE", "Nhân viên ép vỉ", "Nhân viên Giải đáp khách hàng",
           "Nhân viên Giám Sát", "Nhân Viên Giám Sát Cảnh Quan", "Nhân viên giao dịch khách hàng",
           "Nhân viên giao hàng", "Nhân viên giao hàng - Chi nhánh Đồng Văn", "Nhân Viên Giao Hàng - Gia Lai - Kv4",
           "Nhân Viên Giao Nhận", "Nhân viên Giao nhận", "Nhân viên giao nhận", "Nhân viên Giặt là",
           "Nhân viên giới thiệu sản phẩm", "Nhân viên giới thiệu sản phẩm  C.23 (Vùng  3)  Bậc II",
           "Nhân viên Giới thiệu sản phẩm - Khu vực Vũng Tàu", "Nhân viên hành chánh", "Nhân viên hành chính",
           "nhân viên hành chính", "Nhân viên hành chính - Làm việc tại TP.HCM (vùng 1)",
           "Nhân viên hành chính nhân sự", "Nhân viên hành lý", "Nhân viên HC - kế toán",
           "Nhân viên hiện trường Dịch vụ hải quan", "Nhân viên hỗ trợ - SP", "Nhân viên hỗ trợ bán hàng",
           "Nhân viên hỗ trợ đại lý", "nhân viên hỗ trợ điều hành gôn", "Nhân Viên Hỗ Trợ Kinh Doanh",
           "Nhân viên Hỗ trợ kỹ thuật", "Nhân viên hoá chất", "Nhân Viên Học Vụ", "nhân viên hợp đồng",
           "Nhân viên Hướng dẫn", "Nhân viên KCS", "Nhân Viên KCS", "Nhân Viên Kcs Thành Phẩm", "Nhân viên Kế hoạch",
           "nhân viên kế toán", "Nhân viên kế toán", "Nhân viên Kế toán", "Nhân viên kế toán  Tổ trưởng văn phòng",
           "Nhân viên Khách Sạn", "Nhân viên khai báo hải quan", "Nhân viên khai thác hàng hóa nội địa",
           "Nhân viên kho", "Nhân viên Kho", "Nhân viên Kho Bảo Hành Điện Máy Xanh",
           "Nhân viên Kho Điện Máy - V03 - Châu Thành A - Hậu Giang", "Nhân viên Kho kiêm Hỗ trợ kỹ thuật",
           "Nhân viên kho phụ liệu", "Nhân viên khuôn", "Nhân viên Kĩ thuật", "Nhân viên kiểm bao đánh ống",
           "Nhân viên kiểm hàng", "nhân viên kiểm phẩm", "Nhân viên Kiểm phẩm kiêm thu mua lúa gạo",
           "Nhân viên kiểm tra chất lượng phòng Lab", "Nhân viên kiểm tra chất lượng sản phẩm", "Nhân viên kinh doanh",
           "Nhân viên Kinh doanh", "nhân viên kinh doanh", "Nhân viên Kinh Doanh",
           "Nhân viên Kinh doanh  Bưu điện huyện Tân Kỳ", "nhân viên kinh doanh  Phòng kinh doanh",
           "Nhân Viên Kinh Doanh Dự án Samsung", "Nhân viên Kỹ  thuật", "Nhân viên kỹ thuật", "Nhân viên Kỹ thuật",
           "nhân viên kỹ thuật", "Nhân viên Kỹ Thuật", "Nhân Viên kỹ thuật", "Nhân viên kỹ thuật - cơ khí - điện",
           "Nhân viên kỹ thuật  Phòng Kỹ thuật - Quản lý Môi trường", "Nhân viên kỹ thuật chăn nuôi",
           "nhân viên kỹ thuật chế tạo", "nhân viên kỹ thuật điện", "Nhân viên kỹ thuật HĐLĐ",
           "Nhân viên kỹ thuật TTVT", "Nhân viên Lái cẩu", "Nhân viên lái máy", "Nhân viên lái xe", "Nhân Viên Lái Xe",
           "nhân viên lái xe", "Nhân viên Lái xe", "Nhân viên lái xe - HN - T90782", "Nhân viên lái xe buÝt",
           "Nhân viên Lái xe buýt", "Nhân Viên Lái Xe Nâng", "Nhân viên lái xe nâng", "Nhân viên lái xe tải",
           "Nhân viên lái xe tải dưới 01 tấn", "nhân viên lái xe taxi", "Nhân viên lái xe taxi",
           "Nhân viên lái xe taxi 04 chỗ", "Nhân viên lái xe xúc đường", "Nhân viên làm phòng",
           "Nhân viên làm việc tại An Giang", "Nhân viên làm việc tại Hải Dương /vùng 2",
           "Nhân viên Làm việc tại Quảng Nam Vùng 2", "Nhân viên Làm vườn", "Nhân viên Lắp đặt", "Nhân viên lập trình",
           "Nhân viên Lễ tân", "Nhân viên lễ tân", "Nhân viên Lobby Bar", "Nhân viên maketing",
           "Nhân viên marketing- chăm sóc khách hàng", "Nhân viên Marketting", "Nhân viên May", "Nhân viên may",
           "Nhân viên mùa vụ Dịch vụ khách hàng", "Nhân viên nấu ăn",
           "Nhân viên nghiệp vụ An Giang  Phòng Nghiệp Vụ ĐTP", "Nhân viên nguyên vật liệu Kho", "Nhân viên nhà ăn",
           "nhân viên nhà hàng", "Nhân viên Nhà hàng", "Nhân viên nhà hàng", "Nhân viên nhận mẫu (Sample reception)",
           "Nhân viên nhân sự", "Nhân viên nhập liệu", "Nhân viên nước cấp-nước thải", "Nhân viên P.NC-PT mẫu",
           "Nhân viên pha chế", "Nhân viên pha trộn sơn trong sản xuất ô tô  xe máy.", "Nhân viên phân liều",
           "Nhân viên phát triển thị trường", "Nhân Viên Phay CNC", "NHÂN VIÊN PHIÊN DỊCH",
           "Nhân viên phòng Chất lượng", "Nhân viên Phòng Chế tạo động cơ", "Nhân viên Phòng Gia công động cơ",
           "Nhân viên Phòng Hành Chính - Nhân Sự", "Nhân viên phòng Hành chính nhân sự",
           "Nhân viên phòng Kế hoạch thị trường", "nhân viên phòng kế toán", "Nhân viên phòng kĩ thuật",
           "Nhân viên Phòng kinh doanh", "Nhân viên phòng Lab", "Nhân viên phòng Logistics",
           "Nhân viên Phòng Quản lý chất lượng", "Nhân viên phòng Sản xuất", "Nhân viên phòng tài vụ",
           "Nhân viên phòng Thiết bị", "Nhân viên phòng thử  nghiệm", "Nhân Viên Phòng Vật tư", "Nhân viên phụ bếp",
           "Nhân viên phụ trách công tác môi trường", "Nhân viên phục vụ", "Nhân viên Phục vụ",
           "Nhân viên phục vụ ẩm thực", "Nhân viên phục vụ ăn uống", "Nhân viên phục vụ bậc 1", "Nhân viên Phục vụ bàn",
           "Nhân viên phục vụ bàn", "Nhân viên Phục vụ Hồ bơi - Khối Lễ Tân  Buồng Phòng  Giải Trí",
           "Nhân viên phục vụ khách hàng", "Nhân viên Phục vụ Nhà hàng", "Nhân viên phục vụ Nhà hàng San Fu Lou",
           "Nhân viên QA", "Nhân viên QC", "Nhân viên QC thành phẩm", "Nhân viên Quận 1  HCM vùng 1",
           "Nhân viên Quan hệ khách hàng", "Nhân viên quản lý bảo trì", "Nhân viên quản lý chất lượng",
           "Nhân viên quản lý vận hành trạm", "Nhân viên quảng bá thương hiệu", "Nhân viên quầy rau quả",
           "Nhân viên S1", "Nhân viên S3", "Nhân viên sản xuất", "Nhân Viên Sản Xuất", "Nhân viên Sản xuất",
           "Nhân viên sản xuất linh kiện điện tử", "Nhân viên sản xuất Module", "Nhân viên sản xuất Phòng gia công",
           "Nhân viên siêu thị", "Nhân Viên Sinh Quản", "Nhân viên sơ cấp", "Nhân viên Sơn Lót 2",
           "Nhân viên sửa bản in", "Nhân viên sữa chữa  bảo dưỡng hệ thống lạnh  kho lạnh",
           "Nhân viên sửa chữa - bảo trì máy may công nghiệp", "Nhân viên sửa chữa bảo trì", "Nhân viên sửa chữa điện",
           "Nhân viên sửa đồ", "Nhân viên tác nghiệp", "Nhân viên Tài xế", "nhân viên tài xế", "Nhân Viên Tạp Dịch",
           "nhân viên tạp vụ", "Nhân viên tạp vụ", "nhân viên Tạp vụ", "Nhân viên thao tác", "nhân viên thao tác",
           "Nhân viên Thao Tác", "Nhân Viên Thao tác", "Nhân viên thay thô sợi con", "nhân viên thay thô sợi con",
           "Nhân viên thị trường", "Nhân viên Thị trường", "Nhân viên thị trường  Phòng kinh doanh",
           "Nhân viên Thiết bị", "Nhân viên thiết bị tin học", "Nhân viên thiết kế", "Nhân viên thợ quầy thịt",
           "Nhân viên thống kê", "nhân viên thống kê. 1.5  bậc 1/15", "Nhân viên thủ kho", "Nhân viên thu ngân",
           "Nhân viên Thu ngân", "Nhân Viên Thu Ngân", "Nhân viên Thu Ngân - Sim Số",
           "Nhân viên Thu Ngân - Sim Số - V02 - T. Quảng Ninh", "Nhân viên thủ quỷ", "Nhân viên thư viện",
           "NHÂN VIÊN THừA HàNH KHO", "Nhân viên thực nghiệm",
           "Nhân Viên ThườNg - Bộ Phận Kinh Doanh Khu Vực Đông Bắc 1", "Nhân viên Tiếp đón Khách Hàng",
           "Nhân viên Tiếp Đón Khách Hàng - V02 - Rạch Giá - Kiên Giang",
           "Nhân viên Tiếp Đón Khách Hàng - V03 - Tỉnh Bình Phước", "Nhân viên Tiếp Đón Khách Hàng - V04 - T. An Giang",
           "Nhân viên tiếp thị", "Nhân Viên Tiếp Thị", "nhân viên tiếp thị", "Nhân viên Tiếp thị qua điện thoại",
           "nhân viên tín dụng", "Nhân Viên tổ Nền", "Nhân viên tòa nhà", "nhân viên tổng đài",
           "Nhân Viên Trade Marketing - Tiền Giang - Kv2", "Nhân viên Trang Trí kiêm Thu ngân - Sim Số",
           "Nhân viên Trang Trí kiêm Thu ngân - Sim Số - V01 - Thủ Dầu Một - Bình Dương", "Nhân viên trị liệu",
           "Nhân viên trợ giảng", "Nhân viên Tư Vấn", "Nhân viên tư vấn",
           "Nhân viên Tư Vấn - V01 -  Thanh Trì - TP. Hà Nội", "Nhân viên Tư Vấn - V01 - Tỉnh Đồng Nai",
           "Nhân viên Tư Vấn - V02 - T. Đồng Nai", "Nhân viên Tư Vấn - V04 - Tỉnh Vĩnh Long",
           "Nhân viên tư vấn bán hàng", "nhân viên tư vấn kinh doanh", "Nhân Viên Tư Vấn Tín Dụng - CD",
           "Nhân viên Tuân thủ trách nhiệm XH", "Nhân viên Tuyển sinh", "Nhân viên vận hành", "Nhân viên vận hành cẩu",
           "Nhân viên vận hành hệ thống lạnh", "Nhân viên vận hành lò hơi", "Nhân viên vận hành máy",
           "Nhân viên vận hành sản xuất", "nhân viên vận hành trạm cấp nước", "Nhân viên văn phòng",
           "NHÂN VIÊN VĂN PHÒNG", "nhân viên văn phòng", "Nhân Viên Văn Phòng", "Nhân viên văn thư",
           "Nhân viên Văn thư", "Nhân Viên Văn Thư", "Nhân viên Vật tư", "Nhân viên vệ sinh bộ phận kéo dài",
           "Nhân viên vệ sinh công cộng", "nhân viên vệ sinh công cộng", "Nhân viên vệ sinh đổ sợi", "Nhân viên VP",
           "Nhân viên xếp ống sợi con", "Nhân viên XN XN Cơ khí", "Nhân viên XNK", "NHÂN VIÊN XỬ LÝ CHI TRẢ",
           "nhân viên xuất hàng", "Nhân viên xuất hàng", "Nhân viên xưởng se sợi", "Nhân viên y tế - văn thư",
           "Nhân Viên Y Tế Trường TH Phường 4B", "nhân viên-101", "Nhân viên-Hà Nội", "Nhân viên-Mercedes Benz VN",
           "Nhân viênPhục vụ", "Nhân viên-TPHCM", "Nhóm I - Vật liệu XD", "nhóm trưởng", "NHóM TRƯởNG",
           "Nhóm Tư vấn Bán hàng", "Nhõn viờn kiểm hàng", "Nobashi III-9", "Nobashi IV-11", "Nữ hộ sinh", "Nữ Hộ sinh",
           "Nữ hộ sinh  trưởng trạm y tế xã đông hưng B", "Nữ hộ sinh trung học", "Nữ hộ sinh Trung học",
           "Nữ hộ sinh Trung học  Nhân viên hợp đồng Khoa Phụ sản", "Nữ hộ sinh trung học phòng HCTH", "NV", "nv",
           "NV Bán Hàng", "NV Bán hàng", "NV bán hàng", "Nv bán hàng tại Cần Thơ", "NV Bán hàng(Tp.Vũng Tàu-BR VT)",
           "NV Bảo Trì/ Se Sợi", "NV bảo vệ", "NV Bảo vệ", "NV Bốc xếp", "NV Chứng từ", "NV chứng từ - Q.Bình Thạnh",
           "NV dệt sợi", "NV giao hàng", "NV Giao nhận-43/5 Lê Đại Hành  P Phước Mỹ  Phan Rang Tháp Chàm Ninh Thuận",
           "NV giới thiệu và tư vấn sản phẩm Microsoft", "NV hành chánh", "NV Hỗ trợ kỹ thuật tại nhà KH", "NV KD",
           "NV Kế toán", "NV kho", "NV Kinh doanh", "NV kinh doanh", "NV KTVT", "NV Kỹ thuật", "NV kỹ thuật",
           "NV Kỷ thuật", "NV Lái xe", "NV lái xe", "NV nghiệp vụ", "NV phụ xe", "NV Phục Vụ quầy Bar", "NV PTTT",
           "NV PV Nhà hàng", "nv qaht", "NV THAO TáC", "NV theo dõi bán hàng", "NV Thị trường", "NV thiết kế",
           "Nv Thống kê", "nv thương  mại", "NV Thương Mại", "NV Tiếp thị", "NV Tiếp Thị",
           "NV trưng bày và giới thiệu SP - làm việc ở An Giang", "NV Tư Vấn - Hưng Yên - R2", "NV Vận hành",
           "NV vận hành", "NV Vận Hành SX II-Đóng Gói", "Nv. Bán Hàng", "NV. Bảo vệ", "NV. Bếp",
           "NV. Sinh Quản  Khu CN Suối Tre  TX. Long Khánh  Đồng Nai", "NV.Phụ Xe", "NVBH", "NVPV nhà ăn trường học",
           "nvvv", "Machinery Operator / Nhân Viên Vận Hành Máy"]

    v13 = ["Tổ phó", "tổ phó", "Tổ Phó", "Tổ Phó  Chuyền 03", "Tổ phó - Giáo viên", "Tổ phó may 5", "Cửa hàng Phó",
           "Cửa hàng Phó -  Vùng 1-Hồ Chí Minh"]

    v14 = ["Chủ Hộ Kinh Doanh", "Chủ Quản Bảo Trì", "Tổ trường", "tổ trưởng", "TỔ TRƯỞNG", "Tổ Trưởng", "Tổ trưởng",
           "Tổ trưởng Bảo vệ", "Tổ trượng bộ phận in và hoàn tất", "Tổ trưởng cạo mủ cao su", "Tổ trưởng cắt",
           "Tổ trưởng chuyên môn", "Tổ Trường Kho Vật Tư", "Tổ trưởng KSC", "tổ trưởng nấu nhôm thau",
           "Tổ trưởng nhà hàng", "Tổ trưởng nhân viên giúp việc nhà", "Tổ trưởng Nhóm Thành Phẩm", "Tổ trưởng phục vụ",
           "tổ trưởng sản xuất", "Tổ trưởng sản xuất", "Tổ trưởng tổ 1-XN1", "TỔ TRƯỞNG TỔ RÁP", "Tôt trưởng khối 5",
           "Phường đội trưởng", "QĐ khai thác mỏ sét - CN nhà máy gạch Vũ Oai", "Quản đốc SX", "Quản lý", "quản lý",
           "Quản lÝ", "Quản lý bán hàng khu vực", "Quản Lý Khâu may", "Quản lý -V01 - TP. Hồ Chí Minh",
           "Chỉ huy trưởng quân sự", "chuyền trưởng", "Cửa hàng trưởng  CNBC Viettel Vĩnh Long",
           "Cửa hàng trưởng VinMart+", "đội  trưởng thi công", "Đội trưởng thi công - Đội Thi công"]

    v15 = ["Trợ Lý", "Trợ lý", "Trợ lý đặt hàng", "Trợ lý hành chính bậc 1", "Trợ lý hậu kỳ", "Trợ Lý May", "Trợ lý QC",
           "Trợ lý Quản lý khách hàng", "Trợ lý TV"]

    v16 = ["Thợ bánh mỳ", "Thợ bảo trì điện", "thợ bê tông", "Thợ cắt", "Thợ chuyên nghiệp quầy Thịt",
           "Thợ cưa kim loại", "Thợ dệt", "Thợ điện", "Thợ Điện", "Thợ điện TĐH", "Thợ đồng", "Thợ gá",
           "Thợ gõ rỉ sơn cơ giới", "thợ hàn", "Thợ hàn", "Thợ Hàn", "Thợ hàn cắt b2", "Thợ Hàn Điện   bậc 3/7",
           "Thợ in phụ", "Thợ lắp", "Thợ lắp ráp", "Thợ mài trụ", "Thợ May", "Thợ may", "thợ may", "Thợ máy",
           "Thợ máy - 3/7", "thợ may cn", "Thợ Máy Công Nghiệp", "Thợ may trong chuyền may", "Thợ mộc",
           "Thợ nguội sửa chữa", "Thợ phụ", "thợ phụ", "Thợ phụ chuyền may công nghiệp", "thợ sắt",
           "Thợ sắt lắp ráp vỏ tàu thủy", "Thợ sửa chữa", "Thợ Sửa chữa", "Thợ ủi", "thợ ủi"]

    v17 = ["kỹ thuật", "Kỹ thuật", "Kỹ thuật phó", "kỹ thuật viên", "Kỹ thuật viên",
           "Kỹ thuật viên  - TTVT Quảng Trạch", "Kỹ thuật viên  Kinh Doanh", "Kỹ thuật viên chẩn đoán hình ảnh",
           "Kỹ thuật viên chính", "Kỹ thuật viên Dây máy", "Kỹ thuật viên Đo đạc địa chính",
           "Kỹ thuật viên Phân tích lỗi và sửa sản phẩm", "kỹ thuật viên sản xuất", "Kỹ thuật viên sơn",
           "Kỹ thuật viên Spa", "Kỹ thuật viên tin học Tổ dịch vụ khách hàng", "Kỹ thuật viên trung cấp",
           "Kỹ thuật viên trung tâm cấp 1", "Kỹ thuật viên vận hành"]

    v18 = ["Lái máy", "Lái máy xúc", "Lái máy xúc đào", "Lái máy xúc lật", "Lái tàu (lái đầu máy xe lửa)", "lái xe",
           "Lái xe", "LáI XE", "Lái Xe", "Lái xe - Máy đào", "Lái xe 46 chỗ", "Lái Xe Ben- Lao động vùng 2",
           "lái xe cẩu", "Lái xe con", "Lái xe con  xe tải dưới 7 5 tấn", "Lái xe hầm", "lái xe hạng B1  B2",
           "lái xe nâng", "Lái xe ô tô tải hạng FC", "LáI xe tải", "Lái xe tải", "Lái xe Taxi", "Lái xe taxi",
           "lái xe Taxi", "Lái xe trộn bê tông (xe tải từ 7 5 tấn đến dưới 20T)", "lái xe trộn dưới 15 tấn",
           "Lái xe trung chuyển Cần Thơ - Vùng 2", "Lái xe vận tải  có trọng tải 20 tấn trở lên",
           "Lái xe vận tải - có trọng tải từ 7 tấn đến dưới 20 tấn", "tài xế", "TàI Xế", "Tài xế", "TÀI XẾ", "Tài Xế",
           "Tài xế  VC"]

    v19 = ["Y sĩ", "y sĩ", "Y sĩ đa khoa", "Y sĩ Đa khoa", "Y sĩ y học cổ truyền", "Y sỹ", "Y Sỹ", "Y sỹ đa khoa",
           "Y sỹ y học cổ truyền", "Y Tá trung học khoa Nhi"]

    v20 = ["Trưởng bộ phận", "Trưởng ca", "Trưởng ca thu phí", "Trưởng công an", "Trưởng công an xã",
           "Trưởng đại diện VP Nha Trang", "Trưởng dây chuyền phòng sản xuất", "Trưởng Khoa Dược",
           "Trưởng nhóm Hỗ trơ thị trường miền Nam", "trưởng phòng", "Trưởng phòng", "Trưởng phòng An toàn Lao động",
           "Trưởng phòng điều dưỡng bệnh viện đa khoa Tân Thạnh", "Trưởng phòng kiểm định", "Trưởng phòng Kỹ Thuật",
           "Trưởng phòng quản lý dự án", "Trưởng phòng sản xuất", "Trưởng phòng Thí nghiệm", "Trưởng phòng Thương mại",
           "Trưởng quầy Bar"]

    v21 = ["Phó trưởng phòng", "Phó trưởng phòng Kinh doanh",
           "Phó trưởng Phòng Kinh doanh Tổng hợp - Điện lực Tương Dương", "Phó trưởng trạm",
           "Phó trưởng trạm y tế Phường Hồng Sơn; Y sỹ hạng IV", "Phó phòng", "Phó Phòng HC NS", "Phó phòng KHKT-VT",
           "Phó phòng kinh doanh"]

    v22 = ["Phó  hiệu trưởng", "Phó Ban Điều hành", "Phó ban thời sự", "Phó BCH quân sự", "Phó Bí thư",
           "Phó bí thư ĐTN", "Phó ca sx sản xuất", "Phó chánh thanh tra", "Phó chủ tịch HDND", "Phó Chủ Tịch HĐND",
           "Phó Chủ Tịch Hội Liên Hiệp Phụ Nữ  Hội Liên Hiệp Phụ Nữ phường", "Phó Chủ tịch Hội Phụ nữ",
           "Phó Chủ Tịch UBND", "Phó Giám Đốc", "Phó giám đốc",
           "Phó Giám đốc Trung tâm Marketing  CSKH và Phát triển Thương hiệu SHB", "Phó Hiệu Trưởng", "Phó quản đốc",
           "Phó quản đốc máy  gầm  điện", "Phó trưởng khoa", "Hiệu phó", "hiệu phó", "Hiệu Phó", "P. Hiệu trưởng",
           "P.GĐốc", "P.Gíam đốc", "PBT. ĐTN"]

    v23 = ["Kĩ sư công trình", "Kĩ thuật viên", "KS-3/8", "KT tổng hợp", "KTV bảo dưỡng", "KTV bảo trì", "KTV cơ dệt",
           "KTV Dây Máy", "KTV Mắc hồ ghép",
           "KTV Sắt hàn. XN Cơ khí ô tô Chuyên dùng An Lạc. 36 Kinh Dương Vương  P.An Lạc A  Q.Bình Tân  TP.HCM.",
           "KTV Sơn", "KTV Thí nghiệm TS", "KTV xét nghiệm", "KTV Xét nghiệm- khoa xét nghiệm", "Kỹ sư", "Kỹ Sư",
           "Kỹ sư  bậc 1/8  hệ số 2 34", "Kỹ sư bảo trì", "Kỹ sư công nghệ kỹ thuật cơ khí 1/8", "kỹ sư điện",
           "Kỹ sư điện", "Kỹ sư điện - phó tổng giám đốc", "kỹ sư hệ thống điện", "Kỹ sư kỹ thuật xây dựng",
           "kỹ sư nhiệt lạnh", "Kỹ sư tập sự bậc 1/9", "Kỹ sư Thiết kế", "kỹ sư tư vấn thiết kế bậc 1/8",
           "Kỹ sư xây dựng", "Kỹ Sư Xây Dựng"]

    v24 = ["K.ĐịNH VIÊN X5", "KCS", "Kế  toán", "Kế toán", "Kế TOáN", "Kế Toán", "kế toán", "Kế toán - Tổ trưởng HC",
           "Kế toán BH - HCM", "Kế toán cửa hàng", "Kế toán tổng hợp", "Kế toán trưởng", "Kế toán viên"]

    v25 = ["lao động hợp đồng", "Lao động phổ thông", "lao động phổ thông", "LAO ĐỘNG PHỔ THÔNG  HC - TỔNG VỤ",
           "Lao động phổ thông (Bộ phận buồng phòng)", "Lao động phổ thông (Tạp vụ)", "LĐ phụ", "LĐPT",
           "LĐPT - Xếp ống máy cuốn", "LĐPT X7LR3 - LắP RáP",
           "Lắp ráp  vận chuyển linh kiện  kiểm tra và đóng gói sản phẩm", "Phụ Bán Hàng", "Phụ bếp", "Phụ gò",
           "Phụ May", "Phụ máy", "Phụ máy Cắt", "Phụ trách Điều Hành_CNVT", "Phụ trách kế toán",
           "Phụ trách Kiểm tra Đảng", "Phụ trách XDĐSVH Cơ Sở", "Phụ trợ gò  XN1", "phụ việc", "Phụ việc",
           "Phụ việc chuẩn bị", "Phụ việc tổ chỉnh", "Phụ Xe", "Phụ xế", "Phụ xế xe tải", "Phụ xếp hàng", "Phục vụ",
           "Phục vụ bàn", "phục vụ bàn", "Phục vụ Sushi"]

    v26 = ['Fill']

    d = {k1: v1, k2: v2, k3: v3, k4: v4, k5: v5, k7: v7, k8: v8, k9: v9, k10: v10, k11: v11, k12: v12, k13: v13,
         k14: v14, k15: v15, k16: v16, k17: v17, k18: v18, k19: v19, k20: v20, k21: v21, k22: v22, k23: v23, k24: v24,
         k25: v25, k26: v26}
    _df['new_maCv'] = 'Other'
    for k, v in d.items():
        pat = '|'.join(v)
        mask = _df['maCv'].str.contains(pat, case=False)

        _df.loc[mask, 'new_ISP'] = k

scaler = StandardScaler()
_ = scaler.fit_transform(X_train)
X_train = pd.DataFrame(_,columns=X_train.columns)

#Oversample
from imblearn.combine import SMOTEENN
X_train, y_train = SMOTEENN(sampling_strategy = 'minority', random_state=42).fit_resample(X_train, y_train)
print("Original training data points:", len(X))
print("SMOTE oversampled data points:",len(X_train))

scaler = StandardScaler()
_ = scaler.fit_transform(X_train)
X_train = pd.DataFrame(_,columns=X_train.columns)

X_train_cv = pd.DataFrame(scaler.transform(X_train_cv),columns=X_train_cv.columns)
X_test = test.drop(columns = 'id')
X_test = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)
X_test = pd.concat([test[['id']], X_test], axis=1)

X_train['label'] = y_train
X_train_cv['label'] = y_train_cv

X_train.to_csv(r'Oversampled Data/train.csv', index = False)
X_train_cv.to_csv(r'Oversampled Data/validate.csv', index = False)
X_test.to_csv(r'Oversampled Data/test.csv', index = False)



# from imblearn.under_sampling import RandomUnderSampler
# rus = RandomUnderSampler( random_state=42, sampling_strategy = {1:5, 0: 5})
# X_train, y_train = rus.fit_resample(X_train, y_train)