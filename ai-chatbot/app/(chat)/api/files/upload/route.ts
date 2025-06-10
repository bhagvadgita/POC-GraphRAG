import { NextResponse } from 'next/server';
import { z } from 'zod';
import { writeFile, mkdir } from 'fs/promises';
import { join } from 'path';
import { auth } from '@/app/(auth)/auth';
import { spawn } from 'child_process';

// Use Blob instead of File since File is not available in Node.js environment
const FileSchema = z.object({
  file: z
    .instanceof(Blob)
    .refine((file) => file.size <= 5 * 1024 * 1024, {
      message: 'File size should be less than 5MB',
    })
    // Update the file type based on the kind of files you want to accept
    .refine((file) => [
      'application/pdf',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'text/plain',
      'text/csv'
    ].includes(file.type), {
      message: 'File type should be PDF, Word, Excel, or text document',
    }),
});

function runIngestionScript(filename: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const scriptPath = join(process.cwd(), 'graphrag', 'data ingestion', 'ingest.py');
    const pythonProcess = spawn('python', [scriptPath, filename]);

    pythonProcess.stdout.on('data', (data) => {
      console.log(`Ingestion stdout: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`Ingestion stderr: ${data}`);
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Ingestion script exited with code ${code}`));
      }
    });
  });
}

export async function POST(request: Request) {
  const session = await auth();

  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  if (request.body === null) {
    return new Response('Request body is empty', { status: 400 });
  }

  try {
    const formData = await request.formData();
    const file = formData.get('file') as Blob;

    if (!file) {
      return NextResponse.json({ error: 'No file uploaded' }, { status: 400 });
    }

    const validatedFile = FileSchema.safeParse({ file });

    if (!validatedFile.success) {
      const errorMessage = validatedFile.error.errors
        .map((error) => error.message)
        .join(', ');

      return NextResponse.json({ error: errorMessage }, { status: 400 });
    }

    // Get filename from formData since Blob doesn't have name property
    const filename = (formData.get('file') as File).name;
    const fileBuffer = await file.arrayBuffer();

    try {
      // Create the directory if it doesn't exist
      const uploadDir = join(process.cwd(), 'graphrag', 'data', 'raw');
      await mkdir(uploadDir, { recursive: true });

      // Write the file to the local directory
      const filePath = join(uploadDir, filename);
      await writeFile(filePath, Buffer.from(fileBuffer));

      // Run the ingestion script
      await runIngestionScript(filename);

      // Return the local file path and other metadata
      return NextResponse.json({
        url: `/data/raw/${filename}`,
        pathname: filename,
        contentType: file.type,
      });
    } catch (error) {
      console.error('Error processing file:', error);
      return NextResponse.json({ error: 'File processing failed' }, { status: 500 });
    }
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to process request' },
      { status: 500 },
    );
  }
}
