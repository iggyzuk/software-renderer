#include <SFML\Graphics.hpp>
#include <SFML\Window.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <cstring>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <map>
#include <utility>
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
    Vector4(float x = 0.0f, float y = 0.0f, float z = 0.0f, float w = 1.0f) {
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
    float x {0.0f};
    float y {0.0f};
    float z {0.0f};
    float w {0.0f};
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

        float halfWidth = (float)width / 2.0f;
        float halfHeight = (float)height / 2.0f;

        matrix[0][0] = halfWidth;
        matrix[1][1] = -halfHeight;
        matrix[3][0] = halfWidth - 0.5f;
        matrix[3][1] = halfHeight - 0.5f;
    }
    void perspective(float fov, float aspectRatio, float zNear, float zFar) {
        identity();

        float tanHalfFOV = (float)tan((fov / 2) * (PI / 180));
        float zRange = zNear - zFar;

        matrix[0][0] = 1.0f / (tanHalfFOV * aspectRatio);
        matrix[1][1] = 1.0f / tanHalfFOV;
        matrix[2][2] = (-zNear -zFar)/zRange;
        matrix[3][2] = 2.0f * zFar * zNear / zRange;
        matrix[2][3] = 1.0f;
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
    float triangleAreaTimesTwo(Vertex b, Vertex c) {
        float x1 = b.position.x - position.x;
        float y1 = b.position.y - position.y;

        float x2 = c.position.x - position.x;
        float y2 = c.position.y - position.y;

        return (x1 * y2 - x2 * y1);
    }
    Vertex lerp(const Vertex& other, float lerpAmt) {
        return Vertex( position.lerp(other.position, lerpAmt),
                       texcoords.lerp(other.texcoords, lerpAmt) );
    }
    bool isInsideViewFrustum() {
        return fabs(position.x) <= fabs(position.w) &&
               fabs(position.y) <= fabs(position.w) &&
               fabs(position.z) <= fabs(position.w);
    }
    float get(int index) {
        if(index == 0) return position.x;
        if(index == 1) return position.y;
        if(index == 2) return position.z;
        if(index == 3) return position.w;
        return 0;
    }

    Vector4 position;
    Vector4 texcoords;
};

std::ostream& operator<<(std::ostream& out, const Vertex& rhs) {
    out << "(" << (int)rhs.position.x << ", " << (int)rhs.position.y << ", " << (int)rhs.position.z << ")";
    return out;
}

class Gradients {
public:
    Gradients(Vertex minYVert, Vertex midYVert, Vertex maxYVert) {

        depth[0] = minYVert.position.z;
        depth[1] = midYVert.position.z;
        depth[2] = maxYVert.position.z;

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

        depthXStep = calcStepX(depth, minYVert, midYVert, maxYVert, oneOverdX);
        depthYStep = calcStepY(depth, minYVert, midYVert, maxYVert, oneOverdY);
    }
    template<typename T>
    inline T calcStepX(T values[], Vertex minYVert, Vertex midYVert, Vertex maxYVert, float oneOverdX) {
        return ((values[1] - values[2]) *
               (minYVert.position.y - maxYVert.position.y) -
               (values[0] - values[2]) *
               (midYVert.position.y - maxYVert.position.y)) * oneOverdX;
    }
    template<typename T>
    inline T calcStepY(T values[], Vertex minYVert, Vertex midYVert, Vertex maxYVert, float oneOverdY) {
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

    float depth[3];
    float depthXStep;
    float depthYStep;
};

class Edge {
public:
    Edge(Gradients gradients, Vertex start, Vertex end, int startIndex) {
        yStart = (int)ceilf(start.position.y);
        yEnd   = (int)ceilf(end.position.y);

        float yDist = end.position.y - start.position.y;
        float xDist = end.position.x - start.position.x;

        float yPrestep = yStart - start.position.y;
        xStep = xDist / yDist;
        x = start.position.x + yPrestep * xStep;
        float xPrestep = x - start.position.x;

        texcoords = gradients.texcoords[startIndex] + (gradients.texcoordsXStep * xPrestep) + (gradients.texcoordsYStep * yPrestep);
        texcoordsStep = gradients.texcoordsYStep + gradients.texcoordsXStep * xStep;

        oneOverZ = gradients.oneOverZ[startIndex] + gradients.oneOverZXStep * xPrestep + gradients.oneOverZYStep * yPrestep;
        oneOverZStep = gradients.oneOverZYStep + gradients.oneOverZXStep * xStep;

        depth = gradients.depth[startIndex] + gradients.depthXStep * xPrestep + gradients.depthYStep * yPrestep;
        depthStep = gradients.depthYStep + gradients.depthXStep * xStep;
    }

    void Step() {
        x         = x + xStep;
        texcoords = texcoords + texcoordsStep;
        oneOverZ  = oneOverZ + oneOverZStep;
        depth     = depth + depthStep;
    }

    float x;
    float xStep;
    int yStart;
    int yEnd;

    Vector4 texcoords;
    Vector4 texcoordsStep;

    float oneOverZ;
    float oneOverZStep;

    float depth;
    float depthStep;
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

        if(destX < 0 || destX >= width || srcX < 0 || srcX >= width) return;
        if(destIndex < 0 || destIndex >= width * height * 4) return;
        if(srcIndex < 0 || srcIndex >= src.width * src.height * 4) return;

        pixels[destIndex]     = src.pixels[srcIndex    ]; // R
        pixels[destIndex + 1] = src.pixels[srcIndex + 1]; // G
        pixels[destIndex + 2] = src.pixels[srcIndex + 2]; // B
        pixels[destIndex + 3] = src.pixels[srcIndex + 3]; // A
    }
    static Bitmap LoadFromFile(const std::string& filename) {
        sf::Texture texture;
        texture.loadFromFile(filename);
        Bitmap bitmap(texture.getSize().x, texture.getSize().y);
        memcpy(bitmap.pixels, texture.copyToImage().getPixelsPtr(), bitmap.width * bitmap.height * 4);
        //bitmap.pixels = (unsigned char*)texture.copyToImage().getPixelsPtr();
        return bitmap;
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

class StarsField {

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
    StarsField(int numStars, float spread, float speed) {
        this->spread = spread;
        this->speed = speed;

        stars.resize(numStars);
        for(int i = 0; i < numStars; ++i) {
            initStar(i);
        }
    }
    ~StarsField() {
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

template<typename T>
T StringToNumber(std::string string) {
    std::istringstream buffer(string);
    T value; buffer >> value;
    return value;
}

class OBJLoader {
public:

    struct OBJIndex {
        unsigned int vertexIndex;
        unsigned int texCoordIndex;
        unsigned int normalIndex;
    };

    struct OBJModel {
        std::vector<Vector4>  vertices;
        std::vector<Vector4>  texCoords;
        std::vector<Vector4>  normals;
        std::vector<OBJIndex> indices;
    };

    struct IndexedModel {
        std::vector<Vector4>      vertices;
        std::vector<Vector4>      texCoords;
        std::vector<Vector4>      normals;
        std::vector<Vector4>      tangents;
        std::vector<unsigned int> indices;
    };

    static OBJModel Load(std::string filename) {

        OBJModel model;

        std::ifstream file(filename, std::ios::binary | std::ios::in);
        if(file.is_open()){
            for(std::string line; std::getline(file,line);){
                std::vector<std::string> tokens = split(line, ' ');
                assert(tokens.size() > 0);

                if(tokens[0] == "#") continue;
                if(tokens[0] == "v") {
                    model.vertices.push_back( Vector4(StringToNumber<float>(tokens[1]),
                                                      StringToNumber<float>(tokens[2]),
                                                      StringToNumber<float>(tokens[3])) );
                }
                else if(tokens[0] == "vt") {
                    model.texCoords.push_back( Vector4(StringToNumber<float>(tokens[1]),
                                                       1.0f - StringToNumber<float>(tokens[2]),
                                                       0.0f,
                                                       1.0f) );
                }
                else if(tokens[0] == "vn") {
                    model.normals.push_back( Vector4(StringToNumber<float>(tokens[1]),
                                                     StringToNumber<float>(tokens[2]),
                                                     StringToNumber<float>(tokens[3]),
                                                     0.0f) );
                }
                else if(tokens[0] == "f") {
                    for(unsigned int i = 1; i < tokens.size(); ++i) {
                        std::vector<std::string> indexTokens = split(tokens[i], '/');

                        OBJIndex index;
                        index.vertexIndex   = StringToNumber<unsigned int>(indexTokens[0]) - 1;
                        index.texCoordIndex = StringToNumber<unsigned int>(indexTokens[1]) - 1;
                        index.normalIndex   = StringToNumber<unsigned int>(indexTokens[2]) - 1;

                        model.indices.push_back(index);
                    }
                }
            }
        }
        file.close();
        return model;
    }

    static IndexedModel ToIndexedModel(const OBJModel& obj) {

        IndexedModel model;
        std::map<unsigned int, unsigned int> indexMap; // OBJModel.indices -> IndexModel.indices
        unsigned int currentVertexIndex = 0;

        for(unsigned int i = 0; i < obj.indices.size(); ++i) {
            OBJIndex currentIndex = obj.indices[i];

            Vector4 currentPosition = obj.vertices[currentIndex.vertexIndex];
            Vector4 currentTexCoord = obj.texCoords[currentIndex.texCoordIndex];
            Vector4 currentNormal   = obj.normals[currentIndex.normalIndex];

            // Check for duplicates O(n^2)
            int previousVertexIndex = -1;
            for(unsigned int j = 0; j < i; ++j) {
                OBJIndex oldIndex = obj.indices[j];

                if(currentIndex.vertexIndex == oldIndex.vertexIndex &&
                   currentIndex.texCoordIndex == oldIndex.texCoordIndex &&
                   currentIndex.normalIndex == oldIndex.normalIndex) {
                    previousVertexIndex = j;
                    break;
                }
            }

            if(previousVertexIndex == -1) {
                indexMap[i] = currentVertexIndex;

                model.vertices.push_back(currentPosition);
                model.texCoords.push_back(currentTexCoord);
                model.normals.push_back(currentNormal);
                model.indices.push_back(currentVertexIndex);
                currentVertexIndex++;
            }
            else {
                model.indices.push_back(indexMap[(unsigned int)previousVertexIndex]);
            }
        }
        return model;
    }

private:

    static std::vector<std::string>& split(const std::string& s, char delim, std::vector<std::string>& elems) {
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, delim)) {
            elems.push_back(item);
        }
        return elems;
    }
    static std::vector<std::string> split(const std::string& s, char delim) {
        std::vector<std::string> elems;
        split(s, delim, elems);
        return elems;
    }
};

class Mesh {
public:
    Mesh(std::string filename) {
        OBJLoader::IndexedModel model = OBJLoader::ToIndexedModel(OBJLoader::Load(filename));

        for(unsigned int i = 0; i < model.vertices.size(); ++i) {
            vertices.push_back(Vertex( model.vertices[i],
                                       model.texCoords[i]) );
        }

        indices.resize(model.indices.size());
        for(unsigned int j = 0; j < model.indices.size(); ++j) {
            indices[j] = model.indices[j];
        }
    }
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
};

class RenderContext : public Bitmap {
public:
    RenderContext(unsigned short width, unsigned short height) :
        Bitmap(width, height) {
        zBuffer = new float[width * height];
        clearDepthBuffer();
    }
    ~RenderContext() {
        delete[] zBuffer;
    }
    void clearDepthBuffer() {
        unsigned int size = width * height;
        for(unsigned int i = 0; i < size; ++i) {
            zBuffer[i] = 1.0f;
        }
    }
    void drawMesh(Mesh mesh, Matrix4 transform, const Bitmap& texture) {
        for(unsigned int i = 0; i < mesh.indices.size(); i += 3) {
            drawTriangle( mesh.vertices[mesh.indices[i    ]].transform(transform),
                          mesh.vertices[mesh.indices[i + 1]].transform(transform),
                          mesh.vertices[mesh.indices[i + 2]].transform(transform),
                          texture );
        }
    }
    void drawTriangle(Vertex v1, Vertex v2, Vertex v3, const Bitmap& texture) {

        bool v1Inside = v1.isInsideViewFrustum();
        bool v2Inside = v2.isInsideViewFrustum();
        bool v3Inside = v3.isInsideViewFrustum();

        if(v1Inside && v2Inside && v3Inside) {
            fillTriangle(v1, v2, v3, texture);
            return;
        }

        if(!v1Inside && !v2Inside && !v3Inside) return;

        std::vector<Vertex> vertices;
        std::vector<Vertex> auxiliaryList;

        vertices.push_back(v1);
        vertices.push_back(v2);
        vertices.push_back(v3);

        if(clipPolygonAxis(vertices, auxiliaryList, 0) &&
           clipPolygonAxis(vertices, auxiliaryList, 1) &&
           clipPolygonAxis(vertices, auxiliaryList, 2)) {

            Vertex initialVertex = vertices[0];
            for(unsigned int i = 1; i < vertices.size() - 1; ++i) {
                fillTriangle(initialVertex, vertices[i], vertices[i + 1], texture);
            }
        }
    }
private:
    bool clipPolygonAxis(std::vector<Vertex>& vertices, std::vector<Vertex>& auxiliaryList, int componentIndex) {
        clipPolygonComponent(vertices, componentIndex, 1.0f, auxiliaryList);
        vertices.clear();

        if(auxiliaryList.empty()) return false;

        clipPolygonComponent(auxiliaryList, componentIndex, -1.0f, vertices);
        auxiliaryList.clear();

        return !vertices.empty();

    }
    void clipPolygonComponent(std::vector<Vertex>& vertices, int componentIndex, float componentFactor, std::vector<Vertex>& result) {
        Vertex previousVertex = vertices[vertices.size() - 1];
        float previousComponent = previousVertex.get(componentIndex) * componentFactor;
        bool previousInside = previousComponent <= previousVertex.position.w;

        for(auto& currentVertex : vertices) {
            float currentComponent = currentVertex.get(componentIndex) * componentFactor;
            bool currentInside = currentComponent <= currentVertex.position.w;

            if(currentInside ^ previousInside) {
                float lerpAmt = (previousVertex.position.w - previousComponent) /
                                ((previousVertex.position.w - previousComponent) -
                                (currentVertex.position.w - currentComponent));

                result.push_back(previousVertex.lerp(currentVertex, lerpAmt));
            }

            if(currentInside) {
                result.push_back(currentVertex);
            }

            previousVertex = currentVertex;
            previousComponent = currentComponent;
            previousInside = currentInside;
        }
    }
    void fillTriangle(Vertex v1, Vertex v2, Vertex v3, const Bitmap& texture) {
        Matrix4 screenspace;
        screenspace.viewport(width, height);

        Vertex minYVert = v1.transform(screenspace).perspectiveDivide();
        Vertex midYVert = v2.transform(screenspace).perspectiveDivide();
        Vertex maxYVert = v3.transform(screenspace).perspectiveDivide();

        if(minYVert.triangleAreaTimesTwo(maxYVert, midYVert) >= 0)
            return;

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

        unsigned int yStart = b.yStart;
        unsigned int yEnd = b.yEnd;

        for (unsigned int j = yStart; j < yEnd; ++j) {
            drawScanLine(*left, *right, j, texture);
            left->Step();
            right->Step();
        }
    }
    void drawScanLine(const Edge& left, const Edge& right, unsigned int j, const Bitmap& texture) {
        int xMin = (int)ceilf(left.x);
        int xMax = (int)ceilf(right.x);

        float xPrestep = xMin - left.x;
        float xDist = right.x - left.x;
        float texCoordXXStep = (right.texcoords.x - left.texcoords.x) / xDist;
        float texCoordYXStep = (right.texcoords.y - left.texcoords.y) / xDist;
        float oneOverZXStep = (right.oneOverZ - left.oneOverZ) / xDist;
        float depthXStep = (right.depth - left.depth) / xDist;

        float texCoordX = left.texcoords.x + texCoordXXStep * xPrestep;
        float texCoordY = left.texcoords.y + texCoordYXStep * xPrestep;
        float oneOverZ = left.oneOverZ + oneOverZXStep * xPrestep;
        float depth = left.depth + depthXStep * xPrestep;

        for(int i = xMin; i < xMax; ++i) {

            int index = i + j * width;

            if(depth < zBuffer[index]) {
                zBuffer[index] = depth;

                float z = 1.0f / oneOverZ;
                int srcX = (int)((texCoordX * z) * (float)(texture.width - 1) + 0.5f);
                int srcY = (int)((texCoordY * z) * (float)(texture.height - 1) + 0.5f);

                copyPixel(i, j, srcX, srcY, texture);
            }

            texCoordX += texCoordXXStep;
            texCoordY += texCoordYXStep;
            oneOverZ += oneOverZXStep;
            depth += depthXStep;
        }
    }

private:
    float* zBuffer;
};

class Animation {
public:
    Animation(int totalFrames, float frameTimeInSeconds) {
        this->currentFrame = 0;
        this->totalFrames = totalFrames;
        this->frameTimeInSeconds = frameTimeInSeconds;
        this->timer = 0.0f;
    }
    void addFrame(Mesh&& mesh) {
        frames.push_back(mesh);
    }
    Mesh& frame() {
        return frames[currentFrame];
    }
    void animate(const float& dt) {
        timer += dt;
        while(timer > frameTimeInSeconds) {
            currentFrame++;
            if(currentFrame >= totalFrames) {
                currentFrame = 0;
            }
            timer -= frameTimeInSeconds;
        }
    }

private:
    std::vector<Mesh> frames;
    unsigned int currentFrame;
    unsigned int totalFrames;
    float frameTimeInSeconds;
    float timer;
};

int main() {

    srand(time(nullptr));

    float scale = 2.0f;

    RenderContext context(1080 / scale, 720 / scale);
    Display display(context, scale);
    StarsField starfield(4096, 64.0f, 4.0f);

    sf::Clock clock;
    float counter = 0.0f;

    Matrix4 projection;
    projection.perspective(90.0f, 800.0f/600.0f, 0.1f, 1000.0f);

    Bitmap texture(16, 16);
    for(int j = 0; j < texture.height; ++j) {
        for(int i = 0; i < texture.width; ++i) {
            bool isLight = (i + j) % 2 == 0;
            if(isLight) {
                texture.setPixel(i, j, Color(0xFEDB00FF));
            } else {
                texture.setPixel(i, j, Color(0xFF9536FF));
            }
        }
    }

    Bitmap marioTex = Bitmap::LoadFromFile("assets/mario.png");
    Bitmap turtleTex = Bitmap::LoadFromFile("assets/turtle.png");

    Mesh portal("assets/portal.obj");
    Mesh mario("assets/mario.obj");
    Mesh box("assets/box.obj");
    Mesh turtle("assets/turtle.obj");

    Animation anim(8, 0.08f);
    anim.addFrame(Mesh("assets/animation/turtle1.obj"));
    anim.addFrame(Mesh("assets/animation/turtle2.obj"));
    anim.addFrame(Mesh("assets/animation/turtle3.obj"));
    anim.addFrame(Mesh("assets/animation/turtle4.obj"));
    anim.addFrame(Mesh("assets/animation/turtle5.obj"));
    anim.addFrame(Mesh("assets/animation/turtle6.obj"));
    anim.addFrame(Mesh("assets/animation/turtle7.obj"));
    anim.addFrame(Mesh("assets/animation/turtle8.obj"));

    while(display.isOpen()) {

        float dt = clock.restart().asSeconds();
        counter += dt;

        context.clear(Color::Grey());
        context.clearDepthBuffer();

        starfield.render(context, dt);

        Matrix4 transform1;
        transform1.translate(-4.0f, 0.0f, 5.0f);
        transform1.rotateY(counter * 50.0f);

        Matrix4 transform2;
        transform2.translate(0.0f, 0.0f, 4.0f);
        transform2.rotateY(counter * 60.0f);
        transform2.translate(0.0f, -1.5f, 0.0f);

        Matrix4 transform3;
        transform3.translate(4.0f, 0.0f, 5.0f);
        transform3.rotateX(counter * 44.0f);
        transform3.rotateY(counter * 44.0f);
        transform3.rotateZ(counter * 44.0f);
        transform2.scale(1.5f, 1.5f, 1.5f);

        anim.animate(dt);
        context.drawMesh(anim.frame(), projection * transform1, turtleTex);
        context.drawMesh(mario, projection * transform2, marioTex);
        context.drawMesh(box, projection * transform3, texture);

        display.draw();
    }

    return 0;
}
