#include <SFML\Graphics.hpp>
#include <SFML\Window.hpp>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#define PI 3.14159265

float random() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
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
    unsigned char* getPixels() {
        return pixels;
    }

    unsigned short width;
    unsigned short height;
    unsigned char* pixels;
};

class Display {
public:
    Display(Bitmap& bitmap) : bitmap(bitmap) {
        window.create(sf::VideoMode(800, 600, 32), "Software Renderer");

        texture.create(bitmap.width, bitmap.height);
        sprite.setTexture(texture);
        sprite.scale(2.0f, 2.0f);
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

        target.clear(Color(12, 12, 12, 255));

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

int main() {

    srand(0);

    Bitmap bitmap(400, 300);
    Display display(bitmap);

    sf::Clock timer;

    Stars3D game(512, 64.0f, 20.0f);

    while(display.isOpen()) {

        float dt = timer.restart().asSeconds();

        game.render(bitmap, dt);

        display.draw();
    }

    return 0;
}
