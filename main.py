import PIL
import PIL.Image
import math
import numpy as np
from collections import namedtuple

cam = {}
cam['hfov'] = math.pi / 3.0
cam['vfov'] = cam['hfov']
cam['hlen'] = math.tan(cam['hfov'] / 2.0)
cam['vlen'] = math.tan(cam['vfov'] / 2.0)

dot = np.dot

def Reflect(v, n):
	return v - 2 * n * np.dot(n, v)
	
def ReflectRay(ray, hit):
	return Ray(hit['location'], Reflect(ray['dir'], hit['normal']))

def Ray(center, dir):
	return {'center': center, 'dir': dir}
	
def Sphere(center, radius, material):
	return {'center': center, 'radius': radius, 'material': material}
	
def Color(red = 0, green = 0, blue = 0):
	return np.array([red, green, blue])
	
def ConstantMaterial(params = {}):
	defaults = {
		'baseColor': Color(0, 0, 1),
		'specular': 0,
		'roughness': 0.1,
		'emissive': Color(0, 0, 0)
	}
	for k, v in defaults.items():
		if(not k in params):
			params[k] = v
	def ApplyMaterial(out):
		for k, v in params.items():
			out[k] = v
	return ApplyMaterial
	
def ColorOutVal(color):
	return (
		int(color[0] * 255),
		int(color[1] * 255),
		int(color[2] * 255),
		int(255))

def RaySky(ray):
	return {
		'depth': math.inf,
		'emissive': Color(0, 0, 1) * max(np.dot(ray['dir'], np.array([0, 1, 0])) * 0.5 + 0.5, 0)
	}

def RaySphere(ray, sph):
	sr = sph['radius']
	rd = ray['dir']
	m = ray['center'] - sph['center']
	b = np.dot(m, rd)
	c = np.dot(m, m) - sr * sr
	if c < 0.0 and b > 0:
		return False
	discr = b*b - c
	if discr < 0.0:
		return False
	t = -b - math.sqrt(discr)
	t = max(t, 0.0)
	q = rd * t
	norm = q + m
	norm = norm / np.linalg.norm(norm)
	hit = {
			'normal':  norm,
			'depth': t,
			'location': q + ray['center']
	}
	if(np.dot(norm, rd) >= 0):
		return False # for rays originating from the surface. Could also solve with an offset. 
	sph['material'](hit)
	return hit
	
def MinDepth(cols):
	mincol = False
	for col in cols:
		if not mincol or (col and col['depth'] < mincol['depth']):
			mincol = col
	return mincol
	
sph1 = Sphere(np.array([-2, 0, 0]), 3.0, ConstantMaterial({'baseColor': Color(0.4, 0.4, 0.9), 'specular': 0.2}))
sph2 = Sphere(np.array([2, 2, 2]), 1.0, ConstantMaterial({'baseColor': Color(0.6, 0, 0), 'specular': 0.9}))
sph3 = Sphere(np.array([0, -15, 0]), 12, ConstantMaterial({'baseColor': Color(.2, .2, .2), 'specular': 1}))
	
def TraceScene(ray):
	return MinDepth([
		RaySky(ray),
		RaySphere(ray, sph1),
		RaySphere(ray, sph2),
		RaySphere(ray, sph3)
	])
	
def Shade(hit, ray, SampleFn):
	rad = Color(0, 0, 0)
	if('emissive' in hit):
		rad = hit['emissive']
	if('normal' in hit):
		if('baseColor' in hit and hit['baseColor'].any()):
			l = np.array([1, 1, 1])
			l = l / np.linalg.norm(l)
			vis = 1
			vishit = TraceScene(Ray(hit['location'], l))
			if(vishit['depth'] < math.inf):
				vis = 0
			rad = rad + vis * hit['baseColor'] * max(0, np.dot(l, hit['normal']))
		if('specular' in hit and hit['specular'] > 0):
			r = ReflectRay(ray, hit)
			rad = rad + hit['specular'] * SampleFn(r)
	return rad

def SampleScene(ray, depth = 3):
	def SampleEnv(ray):
		return SampleScene(ray, depth - 1)
	if(depth > 0):
		return Shade(TraceScene(ray), ray, SampleEnv)
	else:
		return Color()

def Trace():
	width = 512
	height = 512
	im = PIL.Image.new('RGBA', (width, height), (0, 0, 128, 255))
	pi = im.load()
	for x in range(0, width):
		for y in range(0, height):
			u = x / width
			v = y / height
			uh = u * 2 - 1
			vh = v * 2 - 1
			vdir = np.array([uh * cam['hlen'], vh * cam['vlen'], -1])
			vdir = vdir / np.linalg.norm(vdir)
			cpos = np.array([0, 0, 10])
			ray = Ray(cpos, vdir)
			t = v
			col = SampleScene(ray)
			pi[x, height - 1 - y] = ColorOutVal(col)
	im.show()

from timeit import Timer
timer = Timer(lambda: Trace())
print(timer.timeit(number=1))
