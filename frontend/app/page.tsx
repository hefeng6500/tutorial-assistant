import { Button } from "@/components/ui/button";
import Link from "next/link";

export default function Home() {
  return (
    <div className="container mx-auto py-16 px-4 text-center">
      <h1 className="text-4xl font-bold mb-6">shadcn/ui 组件演示</h1>
      <p className="text-lg text-muted-foreground mb-8">
        这是一个展示 shadcn/ui 组件库的演示项目
      </p>
      <div className="flex gap-4 justify-center">
        <Button asChild>
          <Link href="/components">查看所有组件</Link>
        </Button>
        <Button variant="outline" asChild>
          <Link href="https://ui.shadcn.com" target="_blank">官方文档</Link>
        </Button>
      </div>
    </div>
  );
}
