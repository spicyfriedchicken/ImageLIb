// MONDAY
// draw_line(p1, p2, color)	Bresenham or anti-aliased line // DONE - NAIVE
// draw_rect(x, y, w, h)	Draws a filled or border-only rectangle // DONE - NAIVE
// draw_circle(x, y, r)	Midpoint or anti-aliased circle // DONE - NAIVE
 
#pragma once

#include "utils.hpp"  // Your image class with getWidth(), getHeight(), data(), etc.
#include <vector>
#include <string>

struct Point {
    int x, y;
};

class Draw {
public:
    static void line(Image& image, Point p1, Point p2, const std::array<uint8_t, 3>& color);
    static void rect(Image& image, int x, int y, int w, int h, const std::array<uint8_t, 3>& color, bool filled = true);
    static void circle(Image& image, int cx, int cy, int r, const std::array<uint8_t, 3>& color, bool filled = false);
    static void text(Image& image, int x, int y, const std::string& text, const std::array<uint8_t, 3>& color);
    static void polygon(Image& image, const std::vector<Point>& points, const std::array<uint8_t, 3>& color, bool filled = true);
};

inline void safeSetPixel(Image& image, int x, int y, const std::array<uint8_t, 3>& color) {
    if (x >= 0 && y >= 0 && x < image.getWidth() && y < image.getHeight()) {
        uint8_t* src = image.data();
        src[y * image.getWidth() + x * image.getChannels() + 0] = color[0];
        src[y * image.getWidth() + x * image.getChannels() + 1] = color[1];
        src[y * image.getWidth() + x * image.getChannels() + 2] = color[2];
    }
}

void Draw::line(Image& image, Point p1, Point p2, const std::array<uint8_t, 3>& color) {
    int x0 = p1.x, y0 = p1.y;
    int x1 = p2.x, y1 = p2.y;

    int dx = abs(x1 - x0), dy = -abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;

    while (true) {
        safeSetPixel(image, x0, y0, color);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

void Draw::rect(Image& image, int x, int y, int w, int h, const std::array<uint8_t, 3>& color, bool filled) {
    if (filled) {
        for (int i = y; i < y + h; ++i)
            for (int j = x; j < x + w; ++j)
                safeSetPixel(image, j, i, color);
    } else {
        line(image, {x, y}, {x + w - 1, y}, color);
        line(image, {x, y}, {x, y + h - 1}, color);
        line(image, {x + w - 1, y}, {x + w - 1, y + h - 1}, color);
        line(image, {x, y + h - 1}, {x + w - 1, y + h - 1}, color);
    }
}

void Draw::circle(Image& image, int cx, int cy, int r, const std::array<uint8_t, 3>& color, bool filled) {
    int x = 0;
    int y = r;
    int d = 1 - r;

    auto plotCircle = [&](int x, int y) {
        safeSetPixel(image, cx + x, cy + y, color);
        safeSetPixel(image, cx - x, cy + y, color);
        safeSetPixel(image, cx + x, cy - y, color);
        safeSetPixel(image, cx - x, cy - y, color);
        safeSetPixel(image, cx + y, cy + x, color);
        safeSetPixel(image, cx - y, cy + x, color);
        safeSetPixel(image, cx + y, cy - x, color);
        safeSetPixel(image, cx - y, cy - x, color);
    };

    while (x <= y) {
        if (filled) {
            for (int i = -x; i <= x; ++i) {
                safeSetPixel(image, cx + i, cy + y, color);
                safeSetPixel(image, cx + i, cy - y, color);
            }
            for (int i = -y; i <= y; ++i) {
                safeSetPixel(image, cx + i, cy + x, color);
                safeSetPixel(image, cx + i, cy - x, color);
            }
        } else {
            plotCircle(x, y);
        }

        if (d < 0) {
            d += 2 * x + 3;
        } else {
            d += 2 * (x - y) + 5;
            y--;
        }
        x++;
    }
}
