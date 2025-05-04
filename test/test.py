import vdifheader as vh

input_filepath = './data/testpeb1_01min.vdif'

# get some headers
headers = vh.get_headers(input_filepath) # as iterator (fast)
headers_list = list(headers) # as list (sticks around)
print(f"Got {len(headers_list)} headers")
for k, v in headers_list[0].to_dict.items():
    print(f"{k}: {v}")

for k, v in headers_list[-1].to_dict.items():
    print(f"{k}: {v}")

# do stuff with a header
first_header = headers_list[0]
last_header = headers_list[-1]
timestamp = first_header.get_timestamp()
print(f"\n\nParsed {len(headers_list)} \nstarting at: {timestamp} \nending at: {last_header.get_timestamp()}")
print(f"Data type: {first_header.data_type}")
print(f"Bits per sample: {first_header.bits_per_sample}")
print(f"Number of channels: {first_header.num_channels}")
print(f"Data frame length: {first_header.data_frame_length}")
first_header.data_frame_number

print(f"Source station: {first_header.get_station_information()}")

# export its values somewhere
first_header.to_csv(output_filepath='./some_input_file_vdif.csv')