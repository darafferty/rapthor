#ifndef IDG_OPENCV_VISUALIZE
#define IDG_OPENCV_VISUALIZE

#include <string>
#include <limits>
#include <cmath>

#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>           // cv::Mat
#include <opencv2/highgui/highgui.hpp>     // cv::imread()

namespace idg {

    template<typename T>
    void display_complex_matrix(
        size_t height,
        size_t width,
        T* data,
        std::string form = "abs",
        std::string colormap = "hot")
    {
        #if defined(DEBUG)
        std::cout << __func__ << "("
                  << form << ","
                  << height << ","
                  << width << ","
                  << data << ")" << std::endl;
        #endif

        // Create float image with values in range [0.0f,1.0f]
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

        // Apply colormap
        auto cmap = cv::COLORMAP_HOT;
        if (colormap == "autumn") {
            cmap = cv::COLORMAP_AUTUMN;
        } else if (colormap == "bone") {
            cmap = cv::COLORMAP_BONE;
        } else if (colormap == "cool") {
            cmap = cv::COLORMAP_COOL;
        } else if (colormap == "hsv") {
            cmap = cv::COLORMAP_HSV;
        } else if (colormap == "jet") {
            cmap = cv::COLORMAP_JET;
        } else if (colormap == "ocean") {
            cmap = cv::COLORMAP_OCEAN;
        } else if (colormap == "pink") {
            cmap = cv::COLORMAP_PINK;
        } else if (colormap == "rainbow") {
            cmap = cv::COLORMAP_RAINBOW;
        } else if (colormap == "spring") {
            cmap = cv::COLORMAP_SPRING;
        } else if (colormap == "summer") {
            cmap = cv::COLORMAP_SUMMER;
        } else if (colormap == "winter") {
            cmap = cv::COLORMAP_WINTER;
        }

        // Note: applyColorMap needs a input of type CV_8U, otherwise it simply does not
        // do anything (bug?)
        cv::Mat tmp_img;
        img.convertTo(tmp_img, CV_8U, 255.0, 0.0);
        cv::Mat color_img;
        cv::applyColorMap(tmp_img, color_img, cmap);

        string frameName = form;
        cv::namedWindow(frameName, CV_WINDOW_AUTOSIZE);
        cv::imshow(frameName, color_img);
        cv::waitKey(100000);

        cv::destroyWindow(frameName);
    }

}  // namespace idg

#endif
