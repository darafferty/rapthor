#ifndef IDG_OPENCV_VISUALIZE
#define IDG_OPENCV_VISUALIZE

#include <string>
#include <limits>
#include <cmath>

#include <opencv2/core/core.hpp>           // cv::Mat
#include <opencv2/highgui/highgui.hpp>     // cv::imread()

namespace idg {

    template<typename T>
    void display_complex_matrix(
        size_t height,
        size_t width,
        T* data,
        std::string form = "abs")
    {
        #if defined(DEBUG)
        std::cout << __func__ << "("
                  << form << ","
                  << height << ","
                  << width << ","
                  << data << ")" << std::endl;
        #endif

        cv::Mat img(height, width, CV_32F);

        if (form == "abs") {
            double max_abs = -std::numeric_limits<double>::infinity();
            for (auto y = 0; y < height; ++y) {
                for (auto x = 0; x < width; ++x) {
                    double val = fabs(data[x + y*width]);
                    if (val > max_abs) max_abs = val;
                }
            }

            for (auto y = 0; y < img.rows; ++y) {
                for (auto x = 0; x < img.cols; ++x) {
                    img.at<float>(y, x) = float( fabs(data[x + y*width]) / max_abs );
                }
            }
        }

        if (form == "log") {
            double max_log = -std::numeric_limits<double>::infinity();
            for (auto y = 0; y < height; ++y) {
                for (auto x = 0; x < width; ++x) {
                    double val = log(fabs(data[x + y*width]) + 1);
                    if (val > max_log) max_log = val;
                }
            }

            for (auto y = 0; y < img.rows; ++y) {
                for (auto x = 0; x < img.cols; ++x) {
                    img.at<float>(y, x) = float( log(fabs(data[x + y*width]) + 1) / max_log );
                }
            }
        }

        string frameName = form;
        cv::namedWindow(frameName, CV_WINDOW_AUTOSIZE);
        cv::imshow(frameName, img);
        cv::waitKey(100000);

        cv::destroyWindow(frameName);
    }

}  // namespace idg

#endif
