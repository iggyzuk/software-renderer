#include <SFML\Graphics.hpp>
#include <SFML\Window.hpp>
#include <iostream>
#include <assert.h>
#include <cstring>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#define PI 3.14159265

template <typename T>
void LOG(const T& value) {
    std::cout << value << std::endl;
}

template <typename U, typename... T>
void LOG(const U& head, const T&... tail) {
    std::cout << head << "; ";
    LOG(tail...);
}

float random() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

class Vector4 {
public:
    Vector4(float x = 0, float y = 0, float z = 0, float w = 1) {
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    }
    float length() const {
        return (float)sqrt(x*x + y*y + z*z + w*w);
    }
    Vector4 normalize() {
        float len = length();
        if(len != 0) {
            x /= len;
            y /= len;
            z /= len;
            w /= len;
        }
        return *this;
    }
    float dot(const Vector4& v) const {
        return x * v.x + y * v.y + z * v.z + w * v.w;
    }
    Vector4 cross(const Vector4& v) {
        float xx = y * v.z - z * v.y;
        float yy = z * v.x - x * v.z;
        float zz = x * v.y - y * v.x;
        return Vector4(xx, yy, zz, 0);
    }
    Vector4 lerp(const Vector4& dest, float factor) {
        return (*this) * (1.0f-factor) + dest * factor;
    }
    Vector4 operator+(const Vector4& v) const {
        return Vector4(x + v.x, y + v.y, z + v.z, w + v.w);
    }
    Vector4 operator-(const Vector4& v) const {
        return Vector4(x - v.x, y - v.y, z - v.z, w - v.w);
    }
    Vector4 operator*(const float factor) const {
        return Vector4(x * factor, y * factor, z * factor, w * factor);
    }
    Vector4 operator/(const float factor) const {
        assert(factor != 0);
        return Vector4(x / factor, y / factor, z / factor, w / factor);
    }
    float x;
    float y;
    float z;
    float w;
};
std::ostream& operator<<(std::ostream& out, const Vector4& rhs) {
    out << "(" << rhs.x << ", " << rhs.y << ", " << rhs.z << ")";
    return out;
}

class Matrix4 {
public:
    Matrix4() {
        identity();
    }
    void identity() {
        for (unsigned int i = 0; i < 4; ++i) {
            for (unsigned int j = 0; j < 4; ++j) {
                matrix[j][i] = (i == j ? 1.0f : 0.0f);
            }
        }
    }
    void viewport(unsigned short width, unsigned short height) {
        identity();
        unsigned short halfWidth = width / 2;
        unsigned short halfHeight = height / 2;
        matrix[0][0] = halfWidth;
        matrix[1][1] = -halfHeight;
        matrix[3][0] = halfWidth;
        matrix[3][1] = halfHeight;
    }
    void perspective(float fov, float aspect, float znear, float zfar) {
        float xymax = znear * tan((float)(fov * PI / 180.0f));
        float ymin = -xymax;
        float xmin = -xymax;

        float width = xymax - xmin;
        float height = xymax - ymin;

        float depth = zfar - znear;
        float q = -(zfar + znear) / depth;
        float qn = -2 * (zfar * znear) / depth;

        float w = 2 * znear / width;
        w = w / aspect;
        float h = 2 * znear / height;

        matrix[0][0] = w;
        matrix[0][1] = 0.0f;
        matrix[0][2] = 0.0f;
        matrix[0][3] = 0.0f;
        matrix[1][0] = 0.0f;
        matrix[1][1] = h;
        matrix[1][2] = 0.0f;
        matrix[1][3] = 0.0f;
        matrix[2][0] = 0.0f;
        matrix[2][1] = 0.0f;
        matrix[2][2] = q;
        matrix[2][3] = -1.0f;
        matrix[3][0] = 0.0f;
        matrix[3][1] = 0.0f;
        matrix[3][2] = qn;
        matrix[3][3] = 0.0f;
    }
    Matrix4 operator*(const Matrix4& mat) {
        Matrix4 m;
        for (unsigned int i = 0; i < 4; ++i) {
            for (unsigned int j = 0; j < 4; ++j) {
                m[i][j] = matrix[0][j] * mat.matrix[i][0] +
                          matrix[1][j] * mat.matrix[i][1] +
                          matrix[2][j] * mat.matrix[i][2] +
                          matrix[3][j] * mat.matrix[i][3];
            }
        }
        return m;
    }
    Vector4 operator*(const Vector4& vec) const {
        float a[4];
        for (unsigned int i = 0; i < 4; ++i) {
            a[i] = matrix[0][i] * vec.x +
                   matrix[1][i] * vec.y +
                   matrix[2][i] * vec.z +
                   matrix[3][i] * vec.w;
        }
        return Vector4(a[0], a[1], a[2], a[3]);
    }
    float* operator[](int a) {
        return matrix[a];
    }
    void translate(float x, float y, float z) {
        Matrix4 m;
        m[3][0] = x;
        m[3][1] = y;
        m[3][2] = z;
        *this = *this * m;
    }
    void rotateX(float angle) {
        Matrix4 m;
        float radian = (angle * ((float)PI / 180.0f));
        float sinus = sin(radian);
        float cosinus = cos(radian);
        m[1][1] = cosinus;
        m[2][2] = cosinus;
        m[1][2] = sinus;
        m[2][1] = -sinus;
        *this = *this * m;
    }

    void rotateY(float angle) {
        Matrix4 m;
        float radian = (angle * ((float)PI / 180.0f));
        float sinus = sin(radian);
        float cosinus = cos(radian);
        m[0][0] = cosinus;
        m[2][2] = cosinus;
        m[0][2] = -sinus;
        m[2][0] = sinus;
        *this = *this * m;
    }

    void rotateZ(float angle) {
        Matrix4 m;
        float radian = (angle * ((float)PI / 180.0f));
        float sinus = sin(radian);
        float cosinus = cos(radian);
        m[0][0] = cosinus;
        m[1][1] = cosinus;
        m[0][1] = sinus;
        m[1][0] = -sinus;
        *this = *this * m;
    }
    void scale(float x, float y, float z) {
        Matrix4 m;
        m[0][0] = x;
        m[1][1] = y;
        m[2][2] = z;
        *this = *this * m;
    }
private:
    float matrix[4][4];
};


class Color {
public:
    Color(unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
        this->r = r;
        this->g = g;
        this->b = b;
        this->a = a;
    }
    Color(unsigned int hex) {
        this->r = (hex >> 24) & 0xFF;
        this->g = (hex >> 16) & 0xFF;
        this->b = (hex >> 8 ) & 0xFF;
        this->a = (hex      ) & 0xFF;
    }
    static Color Random() {
        return Color(random()*0xFF, random()*0xFF, random()*0xFF, 0xFF);
    }
    static Color White() {
        return Color(0xFFFFFFFF);
    }
    static Color Grey() {
        return Color(0x151515FF);
    }
    static Color Red() {
        return Color(0xFF0000FF);
    }
    static Color Green() {
        return Color(0x00FF00FF);
    }
    static Color Blue() {
        return Color(0x0000FFFF);
    }
    Vector4 asVector4() const {
        return Vector4((float)r,(float)g,(float)b,(float)a);
    }

    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
};

class Vertex {
public:
    Vertex(Vector4 position, Vector4 texcoords) :
        position(position),
        texcoords(texcoords) {
    }
    float triangleAreaTimesTwo(Vertex b, Vertex c) {
        float x1 = b.position.x - position.x;
        float y1 = b.position.y - position.y;

        float x2 = c.position.x - position.x;
        float y2 = c.position.y - position.y;

        return (x1 * y2 - x2 * y1);
    }
    static Vertex Random() {
        Vector4 v((random() - 0.5f) * 2.0f, (random() - 0.5f) * 2.0f, 1.0f, 1.0f);
        return Vertex(v, Color::Random().asVector4());
    }
    Vertex transform(Matrix4& mat) {
        return Vertex(mat * position, texcoords);
    }
    Vertex perspectiveDivide() const {
        return Vertex(Vector4(position.x/position.w, position.y/position.w, position.z/position.w, position.w), texcoords);
    }

    Vector4 position;
    Vector4 texcoords;
};

std::ostream& operator<<(std::ostream& out, const Vertex& rhs) {
    out << "(" << (int)rhs.position.x << ", " << (int)rhs.position.y << ")";
    return out;
}

class Gradients {
public:
    Gradients(Vertex minYVert, Vertex midYVert, Vertex maxYVert) {

        oneOverZ[0] = 1.0f / minYVert.position.w;
        oneOverZ[1] = 1.0f / midYVert.position.w;
        oneOverZ[2] = 1.0f / maxYVert.position.w;

        texcoords[0] = minYVert.texcoords * oneOverZ[0];
        texcoords[1] = midYVert.texcoords * oneOverZ[1];
        texcoords[2] = maxYVert.texcoords * oneOverZ[2];

        float oneOverdX =
            1.0f /
            (((midYVert.position.x - maxYVert.position.x) *
            (minYVert.position.y - maxYVert.position.y)) -
            ((minYVert.position.x - maxYVert.position.x) *
            (midYVert.position.y - maxYVert.position.y)));
        float oneOverdY = -oneOverdX;

        texcoordsXStep = calcStepX(texcoords, minYVert, midYVert, maxYVert, oneOverdX);
        texcoordsYStep = calcStepY(texcoords, minYVert, midYVert, maxYVert, oneOverdY);

        oneOverZXStep = calcStepX(oneOverZ, minYVert, midYVert, maxYVert, oneOverdX);
        oneOverZYStep = calcStepY(oneOverZ, minYVert, midYVert, maxYVert, oneOverdY);
    }
    template<typename T>
    T calcStepX(T values[], Vertex minYVert, Vertex midYVert, Vertex maxYVert, float oneOverdX) {
        return ((values[1] - values[2]) *
               (minYVert.position.y - maxYVert.position.y) -
               (values[0] - values[2]) *
               (midYVert.position.y - maxYVert.position.y)) * oneOverdX;
    }
    template<typename T>
    T calcStepY(T values[], Vertex minYVert, Vertex midYVert, Vertex maxYVert, float oneOverdY) {
        return ((values[1] - values[2]) *
               (minYVert.position.x - maxYVert.position.x) -
               (values[0] - values[2]) *
               (midYVert.position.x - maxYVert.position.x)) * oneOverdY;
    }

    Vector4 texcoords[3];
    Vector4 texcoordsXStep;
    Vector4 texcoordsYStep;

    float oneOverZ[3];
    float oneOverZXStep;
    float oneOverZYStep;
};

class Edge {
public:
    Edge(Gradients gradients, Vertex start, Vertex end, int startIndex) {
        yStart = (int)ceil(start.position.y);
        yEnd   = (int)ceil(end.position.y);

        float yDist = end.position.y - start.position.y;
        float xDist = end.position.x - start.position.x;

        float yPrestep = yStart - start.position.y;

        xStep = xDist/yDist;
        x = start.position.x + yPrestep * xStep;

        float xPrestep = x - start.position.x;

        texcoords = gradients.texcoords[startIndex] + gradients.texcoordsYStep * yPrestep + gradients.texcoordsXStep * xPrestep;
        texcoordsStep = gradients.texcoordsYStep + gradients.texcoordsXStep * xStep;

        oneOverZ = gradients.oneOverZ[startIndex];
        oneOverZStep = gradients.oneOverZYStep + gradients.oneOverZXStep * xStep;
    }

    void Step() {
        x     = x + xStep;
        texcoords = texcoords + texcoordsStep;
        oneOverZ = oneOverZ + oneOverZStep;
    }

    float x;
    float xStep;
    int yStart;
    int yEnd;

    Vector4 texcoords;
    Vector4 texcoordsStep;

    float oneOverZ;
    float oneOverZStep;
};

class Bitmap {
public:
    Bitmap(unsigned short width, unsigned short height) {
        this->width = width;
        this->height = height;
        this->pixels = new unsigned char[width * height * 4];
    }
    ~Bitmap() {
        delete[] pixels;
    }
    void clear(Color color) {
        for (unsigned int x = 0; x < width; ++x) {
            for (unsigned int y = 0; y < height; ++y) {
                setPixel(x, y, color);
            }
        }
    }
    void setPixel(unsigned int x, unsigned int y, Color color) {
        int index = (x + y * width) * 4;
        if(index < 0 || index >= width * height * 4) return;

        pixels[index]     = color.r; // R
        pixels[index + 1] = color.g; // G
        pixels[index + 2] = color.b; // B
        pixels[index + 3] = color.a; // A
    }
    void copyPixel(unsigned int destX, unsigned int destY, unsigned int srcX, unsigned int srcY, const Bitmap& src) {
        int destIndex = (destX + destY * width) * 4;
        int srcIndex = (srcX + srcY * src.width) * 4;

        if(destIndex < 0 || destIndex >= width * height * 4) return;
        if(srcIndex < 0 || srcIndex >= src.width * src.height * 4) return;

        pixels[destIndex]     = src.pixels[srcIndex    ]; // R
        pixels[destIndex + 1] = src.pixels[srcIndex + 1]; // G
        pixels[destIndex + 2] = src.pixels[srcIndex + 2]; // B
        pixels[destIndex + 3] = src.pixels[srcIndex + 3]; // A
    }

    unsigned short width;
    unsigned short height;
    unsigned char* pixels;
};

class Display {
public:
    Display(Bitmap& bitmap, float scale) : bitmap(bitmap) {
        window.create(sf::VideoMode(bitmap.width * scale, bitmap.height * scale, 32), "Software Renderer");

        texture.create(bitmap.width, bitmap.height);
        sprite.setTexture(texture);
        sprite.scale(scale, scale);
    }
    void draw() {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        texture.update(bitmap.pixels);
        window.draw(sprite);
        window.display();
    }
    bool isOpen() {
        return window.isOpen();
    }
private:
    sf::RenderWindow window;
    sf::Texture      texture;
    sf::Sprite       sprite;
    Bitmap&          bitmap;
};

class Stars3D {

    class Star {
    public:
        Star(unsigned int id, float x, float y, float z, Color color) : color(color) {
            this->id = id;
            this->x  = x;
            this->y  = y;
            this->z  = z;
        }
        unsigned int id;
        float x,y,z;
        Color color;
    };

public:
    Stars3D(int numStars, float spread, float speed) {
        this->spread = spread;
        this->speed = speed;

        stars.resize(numStars);
        for(int i = 0; i < numStars; ++i) {
            initStar(i);
        }
    }
    ~Stars3D() {
        for(unsigned int i = 0; i < stars.size(); ++i) {
            delete stars[i];
        }
        stars.empty();
    }
    void initStar(int index) {
        float x = 2 * (random() - 0.5f) * spread;
        float y = 2 * (random() - 0.5f) * spread;
        float z = (random() + 0.001f) * spread;
        stars[index] = new Star(index, x, y, z, Color::Random());
    }
    void render(Bitmap& target, const float dt) {

        float halfFOV = tan((130.0f / 2.0f) * (PI / 180.0f));

        unsigned int halfWidth = target.width / 2;
        unsigned int halfHeight = target.height / 2;

        target.clear(Color::Grey());

        for(auto& star : stars) {
            star->z -= speed * dt;
            if(star->z <= 0.0f) initStar(star->id);

            int x = (star->x / (star->z * halfFOV)) * halfWidth + halfWidth;
            int y = (star->y / (star->z * halfFOV)) * halfHeight + halfHeight;

            if(x <= 0 || x > target.width || y <= 0 || y > target.height) {
                initStar(star->id);
            } else {
                target.setPixel(x, y, star->color);
            }


        }
    }
private:
    float speed;
    float spread;

    std::vector<Star*> stars;
};

class RenderContext : public Bitmap {
public:
    RenderContext(unsigned short width, unsigned short height) :
        Bitmap(width, height) {
    }
    ~RenderContext() {
    }
    void fillTriangle(Vertex v1, Vertex v2, Vertex v3, const Bitmap& texture) {
        Matrix4 screenspace;
        screenspace.viewport(width, height);

        Vertex minYVert = v1.transform(screenspace).perspectiveDivide();
        Vertex midYVert = v2.transform(screenspace).perspectiveDivide();
        Vertex maxYVert = v3.transform(screenspace).perspectiveDivide();

        if(maxYVert.position.y < midYVert.position.y) {
            Vertex temp = maxYVert;
            maxYVert = midYVert;
            midYVert = temp;
        }

        if(midYVert.position.y < minYVert.position.y) {
            Vertex temp = midYVert;
            midYVert = minYVert;
            minYVert = temp;
        }

        if(maxYVert.position.y < midYVert.position.y) {
            Vertex temp = maxYVert;
            maxYVert = midYVert;
            midYVert = temp;
        }

        scanTriangle(minYVert, midYVert, maxYVert, minYVert.triangleAreaTimesTwo(maxYVert, midYVert) >= 0, texture);
    }
private:
    void scanTriangle(Vertex minYVert, Vertex midYVert, Vertex maxYVert, bool handedness, const Bitmap& texture) {
        Gradients gradiends (minYVert, midYVert, maxYVert);
        Edge topToBottom    (gradiends, minYVert, maxYVert, 0);
        Edge topToMiddle    (gradiends, minYVert, midYVert, 0);
        Edge middleToBottom (gradiends, midYVert, maxYVert, 1);

        scanEdges(topToBottom, topToMiddle, handedness, texture);
        scanEdges(topToBottom, middleToBottom, handedness, texture);
    }
    void scanEdges(Edge& a, Edge& b, bool handedness, const Bitmap& texture) {
        Edge* left = &a;
        Edge* right = &b;

        if(handedness) {
            Edge* temp = left;
            left = right;
            right = temp;
        }

        int yStart = b.yStart;
        int yEnd = b.yEnd;

        for (int j = yStart; j < yEnd; ++j) {
            drawScanLine(*left, *right, j, texture);
            left->Step();
            right->Step();
        }
    }
    void drawScanLine(const Edge& left, const Edge& right, unsigned int j, const Bitmap& texture) {
        int xMin = (int)ceil(left.x);
        int xMax = (int)ceil(right.x);

        Vector4 minTexcoords = left.texcoords;
        Vector4 maxTexcoords = right.texcoords;

        Vector4 minZ = Vector4(0.0f, 0.0f, left.oneOverZ, 0.0f);
        Vector4 maxZ = Vector4(0.0f, 0.0f, right.oneOverZ, 0.0f);

        float lerpAmt = 0.0f;
        float leftStep = 1.0f/(xMax-xMin);

        for(int i = xMin; i < xMax; ++i) {
            Vector4 texcoords = minTexcoords.lerp(maxTexcoords, lerpAmt);
            Vector4 oneOverZ = minZ.lerp(maxZ, lerpAmt);

            float z = 1.0f / oneOverZ.z;

            int srcX = (int)((texcoords.x * z) * texture.width);
            int srcY = (int)((texcoords.y * z) * texture.height);

            copyPixel(i, j, srcX, srcY, texture);

            lerpAmt += leftStep;
        }
    }
};

int main() {

    srand(time(nullptr));

    float scale = 2.0f;

    RenderContext context(1080 / scale, 720 / scale);
    Display display(context, scale);
    Stars3D game(4096, 64.0f, 4.0f);

    sf::Clock clock;
    float counter = 0.0f;

    Matrix4 projection;
    projection.perspective(45.0f, 800.0f/600.0f, 0.1f, 1000.0f);

    Vertex v1 = Vertex(Vector4(-1.0f, -1.0f, 0.0f), Vector4(0.0f, 0.0f));
    Vertex v2 = Vertex(Vector4(0.0f, 1.0f, 0.0f),   Vector4(0.5f, 1.0f));
    Vertex v3 = Vertex(Vector4(1.0f, -1.0f, 0.0f),  Vector4(1.0f, 0.0f));

    Bitmap texture(16, 16);
    for(int j = 0; j < texture.height; ++j) {
        for(int i = 0; i < texture.width; ++i) {
            bool isLight = (i + j) % 2 == 0;
            if(isLight) {
                texture.setPixel(i, j, Color::White());
            } else {
                texture.setPixel(i, j, Color::Red());
            }
        }
    }

    while(display.isOpen()) {

        float dt = clock.restart().asSeconds();
        counter += dt;

        game.render(context, dt);

        Matrix4 transform;
        transform.translate(0.0f, 0.0f, -5.0f);
        transform.rotateZ(counter * 80.0f);
        transform.rotateY(counter * 80.0f);
        transform.rotateZ(counter * 80.0f);
        transform.scale(2.0f, 2.0f, 2.0f);

        Matrix4 mvp = projection * transform;

        context.fillTriangle(
                v1.transform(mvp),
                v2.transform(mvp),
                v3.transform(mvp),
                texture);

        display.draw();
    }

    return 0;
}
