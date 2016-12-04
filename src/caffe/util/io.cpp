#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>
#include <cstdio>
#include <unordered_map>
#include <utility>
#include <cstdlib>
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#ifdef USE_GDAL
#include "gdal_priv.h"
#include "ogrsf_frmts.h" //for ogr
#include "gdal_alg.h"  //for GDALPolygonize
#include "cpl_conv.h" //for CPLMalloc()
#endif

#ifdef USE_MOG
#include "caffe/util/mog.h"
#endif

#include <map>

#ifdef USE_MEMCACHED
#include "MemcachedClient.h"
#endif

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

typedef struct PACKED {
  char name[20];
  unsigned int total;
  char name2[20];
  unsigned int free;
} MEM_OCCUPY;


namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextStream(FileInputStream * input, NetParameter* proto) {
  using google::protobuf::FieldDescriptor;
#define COPY_NON_REPEATED(method) \
  reflection->Set##method(proto, field, reflection->Get##method(temp_proto, field));
#define COPY_NON_REPEATED_MESSAGE(method) \
  reflection->Mutable##method(proto, field)->CopyFrom(reflection->Get##method(temp_proto, field))
#define COPY_REPEATED(method) \
  reflection->Add##method(proto, field, reflection->GetRepeated##method(temp_proto, field, iter->second.index));
#define COPY_REPEATED_MESSAGE(method) \
  reflection->Add##method(proto, field)-> \
          CopyFrom(reflection->GetRepeated##method(temp_proto, field, iter->second.index));
#define CASE_TYPE(cpptype, method, command) \
  case FieldDescriptor::CPPTYPE_##cpptype:\
  {    command(method); break;  }
#define SWITCH_TYPE(command, command2)                       \
  switch(field->cpp_type()) {                                \
  CASE_TYPE(INT32,    Int32,    command)                     \
  CASE_TYPE(BOOL,     Bool,     command)                     \
  CASE_TYPE(STRING,   String,   command)                     \
  CASE_TYPE(MESSAGE,  Message,  command2)                    \
  default: LOG(FATAL) << field->cpp_type() <<                \
  " not supported yet, please modify src/caffe/util/io.cpp"; \
  }
  struct field_index {
    const FieldDescriptor* field;
    int index;
    field_index(const FieldDescriptor *f, int i) {
      field = f;
      index = i;
    }
  };
  struct compare {
    bool operator()(const google::protobuf::TextFormat::ParseLocation & k1,
                    const google::protobuf::TextFormat::ParseLocation & k2) {
      if (k1.line == k2.line) return k1.column < k2.column;
      return k1.line < k2.line;
    }
  };
  typedef pair<google::protobuf::TextFormat::ParseLocation, field_index> location;
  typedef map<google::protobuf::TextFormat::ParseLocation, field_index, compare> field_map;

  NetParameter temp_proto;
  google::protobuf::TextFormat::ParseInfoTree tree;
  google::protobuf::TextFormat::Parser parser;
  parser.WriteLocationsTo(&tree);
  bool success = parser.Parse(input, &temp_proto);
  if (!success) return false;
  //if (!temp_proto.include_size()) {
  //  proto->MergeFrom(temp_proto);
  //  return true;
  //}
  const google::protobuf::Descriptor* descriptor = temp_proto.GetDescriptor();
  const google::protobuf::Reflection* reflection = proto->GetReflection();
  field_map map_field;
  for (int i = 0; i < descriptor->field_count(); ++i) {
    const FieldDescriptor* field = descriptor->field(i);
    if (!field->is_repeated()) {
      CHECK(!reflection->HasField(*proto, field)) <<
          field->name() << " has been set before";
      if (reflection->HasField(temp_proto, field))
        SWITCH_TYPE(COPY_NON_REPEATED, COPY_NON_REPEATED_MESSAGE);
    } else {
      for (int j = 0; j < reflection->FieldSize(temp_proto, field); ++j) {
        map_field.insert(location(tree.GetLocation(field, j), field_index(field, j)));
      }
    }
  }
  for (field_map::iterator iter = map_field.begin(); iter != map_field.end(); ++iter) {
    const FieldDescriptor* field = iter->second.field;
    if (field->name() == "include") {
      const char *filename = reflection->GetRepeatedString(temp_proto, field, iter->second.index).c_str();
      int fd = open(filename, O_RDONLY);
      CHECK_NE(fd, -1) << "File not found: " << filename;
      FileInputStream* input = new FileInputStream(fd);
      if (!ReadProtoFromTextStream(input, proto))
        return false;
    } else {
      SWITCH_TYPE(COPY_REPEATED, COPY_REPEATED_MESSAGE)
    }
  }
  return true;
}

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = false;
  if (proto->GetTypeName() == "caffe.NetParameter")
    success = ReadProtoFromTextStream(input, reinterpret_cast<NetParameter*>(proto));
  else
    success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

void get_memoccupy (MEM_OCCUPY *mem) {
  FILE *fd;
  char buff[256];
  MEM_OCCUPY *m;
  m = mem;
  string meminfo_file = "/proc/meminfo";
  fd = fopen (meminfo_file.c_str(), "r");
  if (fgets (buff, sizeof(buff), fd) != NULL)
    sscanf (buff, "%s %u %s", m->name, &m->total, m->name2);
  if (fgets (buff, sizeof(buff), fd) != NULL)
    sscanf (buff, "%s %u %s", m->name2, &m->free, m->name2);
  fclose(fd);
}

cv::Mat ReadImageToCVMat(const string& filename,
      const int height, const int width, const bool is_color){
  return ReadImageToCVMat(filename, height, width, is_color, false, false, NULL, NULL);
}
cv::Mat ReadImageToCVMat(const string& filename,
      const int height, const int width, const bool is_color, const bool is_label,
      int* img_height, int* img_width) {
#if 0
  // no cache
  cv::Mat cv_img, cv_img_origin;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
                      CV_LOAD_IMAGE_GRAYSCALE);
  cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
#endif
#ifdef USE_MOG
  // LOG(ERROR) << "Read from MOG " << filename;
  char resolved_path[100000];
  memset(resolved_path, 0, 100000);
  realpath(filename.c_str(), resolved_path);
  // LOG(ERROR) << "Resolved path " << resolved_path;

  static st::mog filecache;
  cv::Mat cv_img, cv_img_origin;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
  std::string buffer;
  int len = filecache.get(std::string(resolved_path), buffer);
  // LOG(ERROR) << "len: " << len;
  // LOG(ERROR) << "buffer.len " << buffer.length();
  if (buffer.length() == 0 ) {
    LOG(ERROR) << "Could not find file in MOG cache " << std::string(resolved_path);
    return cv_img_origin;
  }
  std::vector<char> vec_buf(buffer.c_str(), buffer.c_str() + buffer.length());
  cv_img_origin = cv::imdecode(vec_buf, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
#elif defined(USE_MEMCACHED)
  char resolved_path[100000];
  memset(resolved_path, 0, 100000);
  realpath(filename.c_str(), resolved_path);
  // read in the list of servers from configure file
  const string server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf";
  MemcachedClient* mclient = MemcachedClient::GetInstance(server_list_config_file);
  vector<char> value;
  size_t value_length = 0;
  value_length = mclient->Get(std::string(resolved_path), value);
  cv::Mat cv_img, cv_img_origin;
  if(value_length == 0){
    LOG(ERROR) << "Could not find file " << filename;
    return cv_img_origin;
  }
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
  cv_img_origin = cv::imdecode(value, cv_read_flag);
#else
  // no cache
  cv::Mat cv_img, cv_img_origin;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
                      CV_LOAD_IMAGE_GRAYSCALE);
  cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  // cache images in the memory
#if 0
  static std::unordered_map<string, string> filecache;
  static double mem_free_per = 1.0;
  static bool flag = false;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img, cv_img_origin;
  if (filecache.find(filename) == filecache.end()) {
    if (mem_free_per >= 0.1) {
      std::streampos size;
      fstream file(filename.c_str(), ios::in | ios::binary | ios::ate);
      if (file.is_open()) {
        size = file.tellg();
        std::string buffer(size, ' ');
        file.seekg(0, ios::beg);
        file.read(&buffer[0], size);
        file.close();
        filecache[filename] = buffer;
        std::vector<char> vec_buf(buffer.c_str(), buffer.c_str() + buffer.size());
        cv_img_origin = cv::imdecode(vec_buf, cv_read_flag);
      } else {
        LOG(ERROR) << "Could not open or find file " << filename;
        return cv_img_origin;
      }
      MEM_OCCUPY mem_stat;
      get_memoccupy((MEM_OCCUPY *)&mem_stat);
      mem_free_per = (double)mem_stat.free / (double)mem_stat.total;
    } else {
      if (!flag) {
        LOG(INFO) << "The number of images cached in the memory is: " << filecache.size();
        flag = true;
      }
      cv_img_origin = cv::imread(filename, cv_read_flag);
    }

    if (!cv_img_origin.data) {
      LOG(ERROR) << "Could not open or find file " << filename;
      return cv_img_origin;
    }
  } else {
    std::vector<char> vec_buf(filecache[filename].c_str(), filecache[filename].c_str() + filecache[filename].size());
    cv_img_origin = cv::imdecode(vec_buf, cv_read_flag);
  }
#endif
#endif

  if (height > 0 && width > 0) {
    int new_width = width;
    int new_height = height;
    // Resize so that the SHORTEST edge has fixed length, set by the none "1" value
    if (height == 1 || width == 1) {
      float length = height > width ? height : width;
      if (cv_img_origin.rows < cv_img_origin.cols) { // height < width
        float scale = length / cv_img_origin.rows;
        new_width = scale * cv_img_origin.cols;
        new_height = length;
      }
      else { // width <= height
        float scale = length / cv_img_origin.cols;
        new_width = length;
        new_height = scale * cv_img_origin.rows;
      }
    }
    // Resize so that the LONGEST edge has fixed length, set by the none "2" value
    if (height == 2 || width == 2) {
      float length = height > width ? height : width;
      if (cv_img_origin.rows > cv_img_origin.cols) { // height > width
        float scale = length / cv_img_origin.rows;
        new_width = scale * cv_img_origin.cols;
        new_height = length;
      }
      else { // width >= height
        float scale = length / cv_img_origin.cols;
        new_width = length;
        new_height = scale * cv_img_origin.rows;
      }
    }
    if (is_label) {
      cv::resize(cv_img_origin, cv_img, cv::Size(new_width, new_height), 0, 0, cv::INTER_NEAREST);
    } else {
      cv::resize(cv_img_origin, cv_img, cv::Size(new_width, new_height));
    }
  } else {
    cv_img = cv_img_origin;
  }
  if (img_height != NULL) {
    *img_height = cv_img.rows;
  }
  if (img_width != NULL) {
    *img_width = cv_img.cols;
  }
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
  const int height, const int width, const bool is_color, const bool is_label, const bool is_rsdata,
  int* img_height, int* img_width) {

  if (is_rsdata) {
#ifdef USE_GDAL
    cv::Mat rs_img;
    cv::Mat rs_img_origin;
    ReadRSImage_GDAL(filename, rs_img_origin);
    if (!rs_img_origin.data) {
      LOG(ERROR) << "Could not open or find file" << filename;
      return rs_img_origin;
    }
    if (height >0 && width > 0) {
       int new_height = height;
       int new_width  = width;
       if (height==1 || width ==1) {
         float length = height > width ? height : width;
         if (rs_img_origin.rows < rs_img_origin.cols) {
           float scale = length/rs_img_origin.rows;
           new_height = length;
           new_width = scale * rs_img_origin.cols;
         } else {
           float scale = length/rs_img_origin.cols;
           new_width = length;
           new_height = scale * rs_img_origin.rows;
         }
       }

      cv::resize(rs_img_origin,rs_img,cv::Size(new_width,new_height));
    } else {
      rs_img = rs_img_origin;
    }
    if (img_height != NULL) {
      *img_height = rs_img.rows;
    }
    if (img_width != NULL) {
      *img_width = rs_img.cols;
    }
    return rs_img;
#else
    LOG(FATAL) << "USE_GDAL flag is not opened in CMakeList.txt";
    cv::Mat rs_img_empty;
    return rs_img_empty;
#endif
  } else {
    return ReadImageToCVMat(filename, height, width, is_color, is_label, img_height, img_width);
  }
}

cv::Mat ReadImageToCVMat(const string& filename,
                         const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
                         const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
  const int height, const int width, const bool is_color, const bool is_label, const bool is_rsdata,
  const bool is_modis_bin, const std::vector<int> channel_list, int* img_height, int* img_width){
  if(is_modis_bin){
    cv::Mat modis;
    cv::Mat modis_origin; 
    if(is_label){
      FILE *fp = fopen(filename.c_str(), "rb");
      CHECK(fp != NULL) << "label file " <<filename.c_str() << " read error!" ;
      unsigned short label_dim[2]; // height,width of bin label
      CHECK_EQ(fread(label_dim,sizeof(unsigned short),2,fp),2);
      int label_dims_ = label_dim[0] * label_dim[1];
      CHECK_GT(label_dims_,0);
      uchar* label_bin = new uchar[label_dims_];
      CHECK_EQ(fread(label_bin,sizeof(uchar),label_dims_,fp),label_dims_);
      fclose(fp);

      cv::Mat LabelMat(label_dim[0],label_dim[1],CV_8UC1);
      uchar* Label_prt = LabelMat.data;
      for(int i = 0; i < label_dim[0]; i++){
        for(int j = 0; j < label_dim[1]; j++){
          Label_prt[i*label_dim[1] + j] = label_bin[i*label_dim[1] + j];
        }
      }
      delete [] label_bin;
      modis_origin = LabelMat;
    }
    else{
      CHECK_GT(channel_list.size(),0);
      FILE *fp = fopen(filename.c_str(), "rb");
      CHECK(fp != NULL) << "data file " <<filename.c_str() << " read error!" ;
      unsigned short data_dim[3]; // height,width,channel of bin data
      CHECK_EQ(fread(data_dim,sizeof(unsigned short),3,fp),3);
      CHECK_GT(data_dim[2],channel_list.back());
      int data_dims_ = data_dim[0] * data_dim[1] * data_dim[2];
      CHECK_GT(data_dims_,0);
      unsigned short* data_bin = new unsigned short[data_dims_];
      CHECK_EQ(fread(data_bin,sizeof(unsigned short),data_dims_,fp),data_dims_);
      fclose(fp);

      int channels = channel_list.size();
      cv::Mat ImgMat(data_dim[0],data_dim[1],CV_16UC(channels));      
      unsigned short* Img_prt = (unsigned short*)ImgMat.data;
      for(int i = 0; i < data_dim[0]; i++){
        for(int j = 0; j < data_dim[1]; j++){
          for(int k = 0; k < channels; k++)
          Img_prt[i*data_dim[1]*channels + j*channels + k] = data_bin[channel_list[k]*data_dim[0]*data_dim[1] + i*data_dim[1] + j];        }
      }      
      delete [] data_bin;
      modis_origin = ImgMat;
    }

    if (height >0 && width > 0) {
      int new_height = height;
      int new_width  = width;
      if (height==1 || width ==1) {
        float length = height > width ? height : width;
        if (modis_origin.rows < modis_origin.cols) {
          float scale = length/modis_origin.rows;
          new_height = length;
          new_width = scale * modis_origin.cols;
        } 
        else {
          float scale = length/modis_origin.cols;
          new_width = length;
          new_height = scale * modis_origin.rows;
        }
      }
      if(is_label){
        cv::resize(modis_origin,modis,cv::Size(new_width,new_height), 0, 0, cv::INTER_NEAREST);
      }
      else{
        cv::resize(modis_origin,modis,cv::Size(new_width,new_height));
      }
    } else {
      modis = modis_origin;
    }
    if (img_height != NULL) {
      *img_height = modis.rows;
    }
    if (img_width != NULL) {
      *img_width = modis.cols;
    }
    return modis;
  }
  else{
    return ReadImageToCVMat(filename, height, width, is_color, is_label, is_rsdata, img_height, img_width);
  }
}

void ReadRSImage_GDAL(const string filename, cv::Mat& imageMat)
{
#ifdef USE_GDAL
  GDALAllRegister();
  GDALDataset* poSrc = (GDALDataset*)GDALOpen(filename.c_str(),GA_ReadOnly);
  int iWidth = poSrc->GetRasterXSize();
  int iHeight = poSrc->GetRasterYSize();
  int iBandCount = poSrc->GetRasterCount();
  int iStartX,iStartY;
  iStartY = 0;
  iStartX = 0;

  GDALDataType  eDataType = poSrc->GetRasterBand(1)->GetRasterDataType();
  unsigned short* poSrcData = new unsigned short[iWidth*iHeight*iBandCount];

  int* pBandMap = new int[iBandCount];
  for(int i = 0; i < iBandCount; i++) {
    pBandMap[i] = i + 1;
  }
  poSrc->RasterIO(GF_Read,iStartX,iStartY,iWidth,iHeight,poSrcData,iWidth,iHeight,eDataType,iBandCount,pBandMap,0,0,0);


  cv::Mat outImgMat(iHeight,iWidth,CV_16UC(4),cv::Scalar::all(1));
  outImgMat.step = iWidth * iBandCount*2;
  unsigned short  tValue;
  for(int i = 0;i <iHeight;i++) {
    unsigned short* rRow = (unsigned short*)(outImgMat.data + i * outImgMat.step);
    for (int j = 0; j < iWidth; j++) {
      for (int k = 0; k < iBandCount; k++) {
        tValue = poSrcData[k*iHeight*iWidth + i * iWidth + j];
        rRow[j*iBandCount + k] = tValue;
      }
    }
  }

  imageMat = outImgMat;

  delete pBandMap;
  delete poSrcData;

  GDALClose((GDALDatasetH)poSrc);
#endif
}


// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.');
  std::string ext = p != fn.npos ? fn.substr(p) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}
bool ReadImageToDatum(const string& filename, const int label,
                      const int height, const int width, const bool is_color,
                      const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
           matchExt(filename, encoding) )
        return ReadFileToDatum(filename, label, datum);
      std::vector<uchar> buf;
      cv::imencode("." + encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                                  buf.size()));
      datum->set_label(label);
      datum->set_encoded(true);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}

bool ReadFileToDatum(const string& filename, const int label,
                     Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in | ios::binary | ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
                      CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}

// Verifies format of data stored in HDF5 file and reshapes blob accordingly.
template <typename Dtype>
void hdf5_load_nd_dataset_helper(
  hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
  Blob<Dtype>* blob) {
  // Verify that the dataset exists.
  CHECK(H5LTfind_dataset(file_id, dataset_name_))
      << "Failed to find HDF5 dataset " << dataset_name_;
  // Verify that the number of dimensions is in the accepted range.
  herr_t status;
  int ndims;
  status = H5LTget_dataset_ndims(file_id, dataset_name_, &ndims);
  CHECK_GE(status, 0) << "Failed to get dataset ndims for " << dataset_name_;
  CHECK_GE(ndims, min_dim);
  CHECK_LE(ndims, max_dim);

  // Verify that the data format is what we expect: float or double.
  std::vector<hsize_t> dims(ndims);
  H5T_class_t class_;
  status = H5LTget_dataset_info(
             file_id, dataset_name_, dims.data(), &class_, NULL);
  CHECK_GE(status, 0) << "Failed to get dataset info for " << dataset_name_;
  CHECK_EQ(class_, H5T_FLOAT) << "Expected float or double data";

  vector<int> blob_dims(dims.size());
  for (int i = 0; i < dims.size(); ++i) {
    blob_dims[i] = dims[i];
  }
  blob->Reshape(blob_dims);
}

template <>
void hdf5_load_nd_dataset<float>(hid_t file_id, const char* dataset_name_,
                                 int min_dim, int max_dim, Blob<float>* blob) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
  herr_t status = H5LTread_dataset_float(
                    file_id, dataset_name_, blob->mutable_cpu_data());
  CHECK_GE(status, 0) << "Failed to read float dataset " << dataset_name_;
}

template <>
void hdf5_load_nd_dataset<double>(hid_t file_id, const char* dataset_name_,
                                  int min_dim, int max_dim, Blob<double>* blob) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
  herr_t status = H5LTread_dataset_double(
                    file_id, dataset_name_, blob->mutable_cpu_data());
  CHECK_GE(status, 0) << "Failed to read double dataset " << dataset_name_;
}


template <>
void hdf5_save_nd_dataset<float>(
  const hid_t file_id, const string& dataset_name, const Blob<float>& blob) {
  hsize_t dims[HDF5_NUM_DIMS];
  dims[0] = blob.num();
  dims[1] = blob.channels();
  dims[2] = blob.height();
  dims[3] = blob.width();
  herr_t status = H5LTmake_dataset_float(
                    file_id, dataset_name.c_str(), HDF5_NUM_DIMS, dims, blob.cpu_data());
  CHECK_GE(status, 0) << "Failed to make float dataset " << dataset_name;
}

template <>
void hdf5_save_nd_dataset<double>(
  const hid_t file_id, const string& dataset_name, const Blob<double>& blob) {
  hsize_t dims[HDF5_NUM_DIMS];
  dims[0] = blob.num();
  dims[1] = blob.channels();
  dims[2] = blob.height();
  dims[3] = blob.width();
  herr_t status = H5LTmake_dataset_double(
                    file_id, dataset_name.c_str(), HDF5_NUM_DIMS, dims, blob.cpu_data());
  CHECK_GE(status, 0) << "Failed to make double dataset " << dataset_name;
}

}  // namespace caffe
