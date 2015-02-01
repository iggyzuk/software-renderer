#include <SFML\Graphics.hpp>
#include <SFML\Window.hpp>
#include <iostream>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#define PI 3.14159265

float random() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

class Vertex {
public:
    Vertex(float x, float y) {
        this->x = x;
        this->y = y;
    }
    float triangleAreaTimesTwo(Vertex b, Vertex c) {
        float x1 = b.x - x;
        float y1 = b.y - y;

        float x2 = c.x - x;
        float y2 = c.y - y;

        return (x1 * y2 - x2 * y1);
    }
    static Vertex Random(float xRange, float yRange) {
        return Vertex(random() * xRange, random() * yRange);
    }
    float x;
    float y;
};

std::ostream& operator<<(std::ostream& out, const Vertex& rhs) {
    out << "(" << (int)rhs.x << ", " << (int)rhs.y << ")";
    return out;
}

class Color {
public:
    Color(unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
        this->r = r;
        this->g = g;
        this->b = b;
        this->a = a;
    }
    static Color Random() {
        return Color(random()*255, random()*255, random()*255, 255);
    }
    static Color Grey() {
        return Color(24, 24, 24, 255);
    }
    static Color White() {
        return Color(255, 255, 255, 255);
    }

    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
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
        if(index >= width * height * 4) return;

        pixels[index]     = color.r; // R
        pixels[index + 1] = color.g; // G
        pixels[index + 2] = color.b; // B
        pixels[index + 3] = color.a; // A
    }

    unsigned short width;
    unsigned short height;
    unsigned char* pixels;
};

class Display {
public:
    Display(Bitmap& bitmap) : bitmap(bitmap) {
        window.create(sf::VideoMode(bitmap.width, bitmap.height, 32), "Software Renderer");

        texture.create(bitmap.width, bitmap.height);
        sprite.setTexture(texture);
        //sprite.scale(2.0f, 2.0f);
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
    RenderContext(unsigned short width, unsigned short height) : Bitmap(width, height) {
        scanBuffer = new unsigned int[height * 2];
    }
    ~RenderContext() {
        delete[] scanBuffer;
    }
    void drawScanBuffer(unsigned int y, unsigned int xMin, unsigned int xMax) {
        scanBuffer[y * 2]     = xMin;
        scanBuffer[y * 2 + 1] = xMax;
    }
    void fillShape(unsigned int yMin, unsigned int yMax) {
        assert(yMin >= 0 && yMax <= height);
        Color randomColor= Color::Random();
        for(unsigned int j = yMin; j < yMax; ++j) {
            unsigned int xMin = scanBuffer[j * 2];
            unsigned int xMax = scanBuffer[j * 2 + 1];

            for(unsigned int i = xMin; i < xMax; ++i) {
                setPixel(i, j, randomColor);
            }
        }
    }
    void fillTriangle(Vertex v1, Vertex v2, Vertex v3) {
        Vertex minYVert = v1;
        Vertex midYVert = v2;
        Vertex maxYVert = v3;

        if(maxYVert.y < midYVert.y) {
            Vertex temp = maxYVert;
            maxYVert = midYVert;
            midYVert = temp;
        }

        if(midYVert.y < minYVert.y) {
            Vertex temp = midYVert;
            midYVert = minYVert;
            minYVert = temp;
        }

        if(maxYVert.y < midYVert.y) {
            Vertex temp = maxYVert;
            maxYVert = midYVert;
            midYVert = temp;
        }

        float area = minYVert.triangleAreaTimesTwo(maxYVert, midYVert);
        unsigned short handedness = area >= 0 ? 1 : 0;

        scanConvertTriangle(minYVert, midYVert, maxYVert, handedness);
        fillShape((unsigned int)minYVert.y, (unsigned int)maxYVert.y);
    }
    void scanConvertTriangle(Vertex minYVert, Vertex midYVert, Vertex maxYVert, unsigned short handedness) {
        scanConvertLine(minYVert, maxYVert, 0 + handedness);
        scanConvertLine(minYVert, midYVert, 1 - handedness);
        scanConvertLine(midYVert, maxYVert, 1 - handedness);
    }
private:
    void scanConvertLine(Vertex minYVert, Vertex maxYVert, unsigned short side) {
        unsigned int yStart = (unsigned int)minYVert.y;
        unsigned int yEnd   = (unsigned int)maxYVert.y;
        unsigned int xStart = (unsigned int)minYVert.x;
        unsigned int xEnd   = (unsigned int)maxYVert.x;

        int yDist = yEnd - yStart;
        int xDist = xEnd - xStart;

        if(yDist <= 0) return;

        float xStep = (float)xDist/(float)yDist;
        float curX = (float)xStart;

        for(unsigned int j = yStart; j < yEnd; ++j) {
            scanBuffer[j * 2 + side] = (unsigned int)curX;
            curX += xStep;
        }
    }

    unsigned int* scanBuffer;
};

int main() {

    srand(0);

    RenderContext context(800, 600);
    Display display(context);
    Stars3D game(1024, 64.0f, 20.0f);

    sf::Clock clock;
    float counter = 0.5f;

    while(display.isOpen()) {

        float dt = clock.restart().asSeconds();
        counter -= dt;

        game.render(context, dt);

        Vertex v1 = Vertex::Random(800.0f, 600.0f);
        Vertex v2 = Vertex::Random(800.0f, 600.0f);
        Vertex v3 = Vertex::Random(800.0f, 600.0f);

        context.fillTriangle(v1,v2,v3);

        display.draw();
    }

    return 0;
}
