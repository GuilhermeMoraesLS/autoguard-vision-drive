-- Create profiles table
CREATE TABLE public.profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  full_name TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
);

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- RLS policies for profiles
CREATE POLICY "Users can view their own profile"
  ON public.profiles FOR SELECT
  USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile"
  ON public.profiles FOR UPDATE
  USING (auth.uid() = id);

CREATE POLICY "Users can insert their own profile"
  ON public.profiles FOR INSERT
  WITH CHECK (auth.uid() = id);

-- Trigger to create profile on user signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = public
AS $$
BEGIN
  INSERT INTO public.profiles (id, full_name)
  VALUES (
    new.id,
    COALESCE(new.raw_user_meta_data->>'full_name', 'Usuário')
  );
  RETURN new;
END;
$$;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Create cars table
CREATE TABLE public.cars (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  brand TEXT NOT NULL,
  model TEXT NOT NULL,
  plate TEXT NOT NULL,
  year INTEGER,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
  UNIQUE(user_id, plate)
);

ALTER TABLE public.cars ENABLE ROW LEVEL SECURITY;

-- RLS policies for cars
CREATE POLICY "Users can view their own cars"
  ON public.cars FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own cars"
  ON public.cars FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own cars"
  ON public.cars FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own cars"
  ON public.cars FOR DELETE
  USING (auth.uid() = user_id);

-- Create authorized_drivers table
CREATE TABLE public.authorized_drivers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  car_id UUID NOT NULL REFERENCES public.cars(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  photo_url TEXT NOT NULL,
  face_encoding TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
);

ALTER TABLE public.authorized_drivers ENABLE ROW LEVEL SECURITY;

-- RLS policies for authorized_drivers (via car ownership)
CREATE POLICY "Users can view drivers for their cars"
  ON public.authorized_drivers FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.cars
      WHERE cars.id = authorized_drivers.car_id
      AND cars.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can insert drivers for their cars"
  ON public.authorized_drivers FOR INSERT
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.cars
      WHERE cars.id = authorized_drivers.car_id
      AND cars.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can update drivers for their cars"
  ON public.authorized_drivers FOR UPDATE
  USING (
    EXISTS (
      SELECT 1 FROM public.cars
      WHERE cars.id = authorized_drivers.car_id
      AND cars.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can delete drivers for their cars"
  ON public.authorized_drivers FOR DELETE
  USING (
    EXISTS (
      SELECT 1 FROM public.cars
      WHERE cars.id = authorized_drivers.car_id
      AND cars.user_id = auth.uid()
    )
  );

-- Create storage bucket for driver photos
INSERT INTO storage.buckets (id, name, public)
VALUES ('driver-photos', 'driver-photos', true);

-- RLS policies for storage
CREATE POLICY "Users can upload photos for their cars"
  ON storage.objects FOR INSERT
  WITH CHECK (
    bucket_id = 'driver-photos' AND
    auth.uid()::text = (storage.foldername(name))[1]
  );

CREATE POLICY "Anyone can view driver photos"
  ON storage.objects FOR SELECT
  USING (bucket_id = 'driver-photos');

CREATE POLICY "Users can update their own photos"
  ON storage.objects FOR UPDATE
  USING (
    bucket_id = 'driver-photos' AND
    auth.uid()::text = (storage.foldername(name))[1]
  );

CREATE POLICY "Users can delete their own photos"
  ON storage.objects FOR DELETE
  USING (
    bucket_id = 'driver-photos' AND
    auth.uid()::text = (storage.foldername(name))[1]
  );