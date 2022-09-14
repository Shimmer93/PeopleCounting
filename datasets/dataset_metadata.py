class SmartCityMetadata:
    img_folder = 'images'
    gt_folder = 'images'
    gt_colname = 'loc'
    gt_prefix = ''
    gt_suffix = ''
    is_grey = False

class UCF_CC_50Metadata:
    img_folder = ''
    gt_folder = ''
    gt_colname = 'annPoints'
    gt_prefix = ''
    gt_suffix = '_ann'
    is_grey = True

class ShanghaiTechAMetadata:
    img_folder = 'img'
    gt_folder = 'annotations'
    gt_colname = 'image_info'
    gt_prefix = 'GT_'
    gt_suffix = ''
    is_grey = False