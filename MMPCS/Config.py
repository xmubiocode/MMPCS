
SEED = 42
FINETUNE_DATASET_BASEPATH = "finetune_dataset"

lr = 5e-5

# [
#     {
#         "id":"xxxx",
#         "depencency":["id1","id2"],
#         "input_files":[
#             "fileName1":{
#                 "type":"minio",
#                 "path":"minio_path"
#             },
#             "fileName2":{
#                 "type":"output_file",
#                 "path":"id1.fileName2"
#             }
#         ],
#         "output_files":[
#             "fileName1",
#             "fileName2"
#         ],
#         "config": [
#             # 还没想好
#         ]
#     },
#     {

#     }
# ]