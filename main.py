import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import math
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from abc import ABC, abstractmethod

# Vector operations
class Vec3:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x, self.y, self.z = x, y, z

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)
        return Vec3(self.x * scalar.x, self.y * scalar.y, self.z * scalar.z)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self):
        l = self.length()
        if l > 0:
            return self / l
        return Vec3(0, 0, 0)

    def reflect(self, normal):
        return self - 2 * self.dot(normal) * normal

    def refract(self, normal, eta):
        cos_i = -self.dot(normal)
        sin_t2 = eta * eta * (1 - cos_i * cos_i)
        if sin_t2 >= 1:
            return None  # Total internal reflection
        cos_t = math.sqrt(1 - sin_t2)
        return eta * self + (eta * cos_i - cos_t) * normal

    def __repr__(self):
        return f"Vec3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

# Color class
class Color(Vec3):
    def clamp(self):
        return Color(
            max(0, min(1, self.x)),
            max(0, min(1, self.y)),
            max(0, min(1, self.z))
        )

    def to_rgb(self):
        clamped = self.clamp()
        return (int(clamped.x * 255), int(clamped.y * 255), int(clamped.z * 255))

# Ray class
@dataclass
class Ray:
    origin: Vec3
    direction: Vec3

    def at(self, t: float) -> Vec3:
        return self.origin + t * self.direction

# Hit record
@dataclass
class HitRecord:
    point: Vec3
    normal: Vec3
    t: float
    front_face: bool
    material: 'Material'

    def set_face_normal(self, ray: Ray, outward_normal: Vec3):
        self.front_face = ray.direction.dot(outward_normal) < 0
        self.normal = outward_normal if self.front_face else -1 * outward_normal

# Abstract Material class
class Material(ABC):
    @abstractmethod
    def scatter(self, ray: Ray, hit_record: HitRecord) -> Tuple[bool, Color, Ray]:
        pass

    @abstractmethod
    def emitted(self, u: float, v: float, point: Vec3) -> Color:
        pass

# Lambertian material
class Lambertian(Material):
    def __init__(self, albedo: Color):
        self.albedo = albedo

    def scatter(self, ray: Ray, hit_record: HitRecord) -> Tuple[bool, Color, Ray]:
        scatter_direction = hit_record.normal + self.random_unit_vector()

        # Catch degenerate scatter direction
        if self.near_zero(scatter_direction):
            scatter_direction = hit_record.normal

        scattered = Ray(hit_record.point, scatter_direction)
        return True, self.albedo, scattered

    def emitted(self, u: float, v: float, point: Vec3) -> Color:
        return Color(0, 0, 0)

    @staticmethod
    def random_unit_vector():
        a = np.random.uniform(0, 2 * math.pi)
        z = np.random.uniform(-1, 1)
        r = math.sqrt(1 - z * z)
        return Vec3(r * math.cos(a), r * math.sin(a), z)

    @staticmethod
    def near_zero(v: Vec3):
        s = 1e-8
        return abs(v.x) < s and abs(v.y) < s and abs(v.z) < s

# Metal material
class Metal(Material):
    def __init__(self, albedo: Color, fuzz: float = 0):
        self.albedo = albedo
        self.fuzz = min(fuzz, 1)

    def scatter(self, ray: Ray, hit_record: HitRecord) -> Tuple[bool, Color, Ray]:
        reflected = ray.direction.normalize().reflect(hit_record.normal)
        reflected = reflected + self.fuzz * Lambertian.random_unit_vector()
        scattered = Ray(hit_record.point, reflected)
        return scattered.direction.dot(hit_record.normal) > 0, self.albedo, scattered

    def emitted(self, u: float, v: float, point: Vec3) -> Color:
        return Color(0, 0, 0)

# Dielectric material
class Dielectric(Material):
    def __init__(self, refractive_index: float):
        self.ir = refractive_index

    def scatter(self, ray: Ray, hit_record: HitRecord) -> Tuple[bool, Color, Ray]:
        attenuation = Color(1.0, 1.0, 1.0)
        refraction_ratio = 1.0 / self.ir if hit_record.front_face else self.ir

        unit_direction = ray.direction.normalize()
        cos_theta = min(-unit_direction.dot(hit_record.normal), 1.0)
        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)

        cannot_refract = refraction_ratio * sin_theta > 1.0

        if cannot_refract or self.reflectance(cos_theta, refraction_ratio) > np.random.random():
            direction = unit_direction.reflect(hit_record.normal)
        else:
            direction = unit_direction.refract(hit_record.normal, refraction_ratio)

        scattered = Ray(hit_record.point, direction)
        return True, attenuation, scattered

    def emitted(self, u: float, v: float, point: Vec3) -> Color:
        return Color(0, 0, 0)

    @staticmethod
    def reflectance(cosine: float, ref_idx: float) -> float:
        # Schlick's approximation
        r0 = (1 - ref_idx) / (1 + ref_idx)
        r0 = r0 * r0
        return r0 + (1 - r0) * pow(1 - cosine, 5)

# Emissive material
class Emissive(Material):
    def __init__(self, emit_color: Color, intensity: float = 1.0):
        self.emit_color = emit_color
        self.intensity = intensity

    def scatter(self, ray: Ray, hit_record: HitRecord) -> Tuple[bool, Color, Ray]:
        return False, Color(0, 0, 0), ray

    def emitted(self, u: float, v: float, point: Vec3) -> Color:
        return self.intensity * self.emit_color

# Abstract Shape class
class Shape(ABC):
    def __init__(self, material: Material):
        self.material = material

    @abstractmethod
    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        pass

# Sphere class
class Sphere(Shape):
    def __init__(self, center: Vec3, radius: float, material: Material):
        super().__init__(material)
        self.center = center
        self.radius = radius

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        half_b = oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = half_b * half_b - a * c

        if discriminant < 0:
            return None

        sqrtd = math.sqrt(discriminant)

        # Find the nearest root that lies in the acceptable range
        root = (-half_b - sqrtd) / a
        if root < t_min or t_max < root:
            root = (-half_b + sqrtd) / a
            if root < t_min or t_max < root:
                return None

        hit_record = HitRecord(
            point=ray.at(root),
            normal=Vec3(),
            t=root,
            front_face=False,
            material=self.material
        )

        outward_normal = (hit_record.point - self.center) / self.radius
        hit_record.set_face_normal(ray, outward_normal)

        return hit_record

# Plane class
class Plane(Shape):
    def __init__(self, point: Vec3, normal: Vec3, material: Material):
        super().__init__(material)
        self.point = point
        self.normal = normal.normalize()

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        denom = self.normal.dot(ray.direction)
        if abs(denom) < 1e-8:
            return None

        t = (self.point - ray.origin).dot(self.normal) / denom
        if t < t_min or t > t_max:
            return None

        hit_record = HitRecord(
            point=ray.at(t),
            normal=Vec3(),
            t=t,
            front_face=False,
            material=self.material
        )

        hit_record.set_face_normal(ray, self.normal)
        return hit_record

# Scene class
class Scene:
    def __init__(self):
        self.objects: List[Shape] = []
        self.lights: List[Shape] = []

    def add(self, obj: Shape):
        self.objects.append(obj)
        if isinstance(obj.material, Emissive):
            self.lights.append(obj)

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        closest_hit = None
        closest_t = t_max

        for obj in self.objects:
            hit_record = obj.hit(ray, t_min, closest_t)
            if hit_record:
                closest_t = hit_record.t
                closest_hit = hit_record

        return closest_hit

# Camera class
class Camera:
    def __init__(self, lookfrom: Vec3, lookat: Vec3, vup: Vec3, vfov: float,
                 aspect_ratio: float, aperture: float, focus_dist: float):
        theta = math.radians(vfov)
        h = math.tan(theta / 2)
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height

        self.w = (lookfrom - lookat).normalize()
        self.u = vup.cross(self.w).normalize()
        self.v = self.w.cross(self.u)

        self.origin = lookfrom
        self.horizontal = focus_dist * viewport_width * self.u
        self.vertical = focus_dist * viewport_height * self.v
        self.lower_left_corner = (self.origin - self.horizontal/2 -
                                  self.vertical/2 - focus_dist * self.w)

        self.lens_radius = aperture / 2

    def get_ray(self, s: float, t: float) -> Ray:
        rd = self.lens_radius * self.random_in_unit_disk()
        offset = self.u * rd.x + self.v * rd.y

        return Ray(
            self.origin + offset,
            (self.lower_left_corner + s * self.horizontal +
             t * self.vertical - self.origin - offset)
        )

    @staticmethod
    def random_in_unit_disk():
        while True:
            p = Vec3(np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0)
            if p.dot(p) < 1:
                return p

# Ray Tracer Engine
class RayTracer:
    def __init__(self, width: int = 800, height: int = 600, samples: int = 100,
                 max_depth: int = 50):
        self.width = width
        self.height = height
        self.samples = samples
        self.max_depth = max_depth
        self.aspect_ratio = width / height

    def ray_color(self, ray: Ray, scene: Scene, depth: int) -> Color:
        if depth <= 0:
            return Color(0, 0, 0)

        hit_record = scene.hit(ray, 0.001, float('inf'))

        if hit_record:
            # Add emitted light
            emitted = hit_record.material.emitted(0, 0, hit_record.point)

            # Scatter ray
            scattered, attenuation, new_ray = hit_record.material.scatter(ray, hit_record)

            if scattered:
                return emitted + attenuation * self.ray_color(new_ray, scene, depth - 1)
            else:
                return emitted

        # Sky gradient
        unit_direction = ray.direction.normalize()
        t = 0.5 * (unit_direction.y + 1.0)
        return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0)

    def render_pixel(self, args):
        i, j, camera, scene = args
        color = Color(0, 0, 0)

        for _ in range(self.samples):
            u = (i + np.random.random()) / (self.width - 1)
            v = (j + np.random.random()) / (self.height - 1)
            ray = camera.get_ray(u, v)
            color = color + self.ray_color(ray, scene, self.max_depth)

        # Gamma correction
        color = color / self.samples
        color = Color(math.sqrt(color.x), math.sqrt(color.y), math.sqrt(color.z))

        return i, j, color.to_rgb()

    def render(self, scene: Scene, camera: Camera) -> np.ndarray:
        print(f"Rendering {self.width}x{self.height} image with {self.samples} samples...")
        start_time = time.time()

        # Prepare pixel coordinates
        pixel_coords = []
        for j in range(self.height - 1, -1, -1):  # Top to bottom
            for i in range(self.width):
                pixel_coords.append((i, j, camera, scene))

        # Use multiprocessing for faster rendering
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(self.render_pixel, pixel_coords)

        # Convert results to image array
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for i, j, rgb in results:
            image[self.height - 1 - j, i] = rgb

        elapsed_time = time.time() - start_time
        print(f"Rendering completed in {elapsed_time:.2f} seconds")

        return image

# Scene Builder
class SceneBuilder:
    @staticmethod
    def create_demo_scene() -> Scene:
        scene = Scene()

        # Materials
        ground = Lambertian(Color(0.5, 0.5, 0.5))
        red_diffuse = Lambertian(Color(0.7, 0.3, 0.3))
        metal = Metal(Color(0.8, 0.8, 0.9), 0.0)
        glass = Dielectric(1.5)
        light = Emissive(Color(1, 1, 1), 5.0)

        # Ground plane
        scene.add(Plane(Vec3(0, -1, 0), Vec3(0, 1, 0), ground))

        # Spheres
        scene.add(Sphere(Vec3(0, 0, -1), 0.5, red_diffuse))
        scene.add(Sphere(Vec3(-1, 0, -1), 0.5, glass))
        scene.add(Sphere(Vec3(-1, 0, -1), -0.45, glass))  # Hollow glass sphere
        scene.add(Sphere(Vec3(1, 0, -1), 0.5, metal))

        # Light source
        scene.add(Sphere(Vec3(0, 2, -1), 0.3, light))

        # Random spheres
        for a in range(-11, 12, 2):
            for b in range(-11, 12, 2):
                choose_mat = np.random.random()
                center = Vec3(a + 0.9 * np.random.random(), 0.2, b + 0.9 * np.random.random())

                if (center - Vec3(4, 0.2, 0)).length() > 0.9:
                    if choose_mat < 0.8:  # Diffuse
                        albedo = Color(np.random.random(), np.random.random(), np.random.random())
                        scene.add(Sphere(center, 0.2, Lambertian(albedo)))
                    elif choose_mat < 0.95:  # Metal
                        albedo = Color(0.5 + 0.5 * np.random.random(),
                                       0.5 + 0.5 * np.random.random(),
                                       0.5 + 0.5 * np.random.random())
                        fuzz = 0.5 * np.random.random()
                        scene.add(Sphere(center, 0.2, Metal(albedo, fuzz)))
                    else:  # Glass
                        scene.add(Sphere(center, 0.2, glass))

        return scene

# Main execution
def main():
    # Create scene
    scene = SceneBuilder.create_demo_scene()

    # Create camera
    lookfrom = Vec3(13, 2, 3)
    lookat = Vec3(0, 0, 0)
    vup = Vec3(0, 1, 0)
    vfov = 20
    aspect_ratio = 16.0 / 9.0
    aperture = 0.1
    focus_dist = 10.0

    camera = Camera(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist)

    # Create ray tracer
    tracer = RayTracer(width=800, height=450, samples=10, max_depth=50)

    # Render
    image = tracer.render(scene, camera)

    # Save image
    img = Image.fromarray(image)
    img.save('raytraced_scene.png')

    # Display
    plt.figure(figsize=(16, 9))
    plt.imshow(image)
    plt.axis('off')
    plt.title('3D Ray Traced Scene')
    plt.tight_layout()
    plt.savefig('raytraced_scene_display.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Ray tracing complete! Image saved as 'raytraced_scene.png'")

if __name__ == "__main__":
    main()